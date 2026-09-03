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

// vk::image_format qualifiers on every RWTexture2D below: without an
// explicit format, Slang emits SPIR-V `OpTypeImage Format=Unknown` +
// the StorageImageWriteWithoutFormat capability. On NVIDIA Vulkan
// drivers (596.x family) that combination silently no-op'd
// OpImageWrite on the full PathTrace kernel, producing all-zero
// output. Each string here matches the engine's VkImage format from
// `src/engine/Engine.cpp` `CreateTexture` calls -- see the bug-fix
// PR for the full mapping table. (See DXC #6153, Slang #5969,
// NVIDIA bug #1080 family.)
[[vk::binding(0, 0), vk::image_format("rgba8")]]   RWTexture2D<float4> output;
[[vk::binding(1, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> accum_hdr;
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
[[vk::binding(6, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> denoise_color;   // RGBA16F per-frame HDR (NOT accumulated)
[[vk::binding(7, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;       // R32F  linear camera-space depth
[[vk::binding(8, 0), vk::image_format("rg16f")]]   RWTexture2D<float2> motion_tex;      // RG16F pixel-space (prev - curr) delta
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
[[vk::binding(9, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> env_map;
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
[[vk::binding(12, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> star_map;
#endif

// Procedural moon surface texture, equirectangular RGBA16F generated
// once at engine init by pt::moon::generateMoonTexture. Sampled in
// moonDisc() to give the disc real mare + crater detail.
#ifdef PT_TARGET_METAL
Texture2D<float4> moon_map;
#else
[[vk::binding(13, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> moon_map;
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
[[vk::binding(16, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> normal_tex;

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
// Declared on BOTH Metal and SPIR-V (NOT #ifdef-wrapped). When the
// declaration was SPIR-V-only, Slang's MSL backend dead-code-eliminated
// it from the kernel signature, which compacted all subsequent texture
// slots by one and silently broke every engine-side BindStorageTexture
// call that landed past it -- see PR #108's regression history.
// Keeping the declaration on both targets preserves Slang's slot
// assignment contract: each declared texture gets a stable MSL index
// in declaration order regardless of backend. The runtime
// `write_albedo_gbuffer` push gate (set to 0 by the engine on Metal
// and on non-OptiX-AOV Vulkan dispatches) is the actual run-or-skip
// switch; the engine binds a placeholder when albedo_tex_id_ is 0 to
// satisfy descriptor validity.
[[vk::binding(17, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> albedo_tex;

// Camera-ray cloud transmittance G-buffer (issue #46 follow-up).
// Single-channel R32F, written by the path tracer's volumetric cloud
// march at the END of the bounce/sample loop, read by the post-
// denoise StarsComposite kernel to attenuate the additive stars +
// sun + moon contribution. Without this gate the composite is "pure
// additive" -- stars / sun / moon punch through clouds at full
// brightness, looking nonphysical. With it, clouds correctly occlude
// the celestials in proportion to the integrated cloud density along
// the camera ray.
//
// Value semantics: trans_eye_c, the running transmittance product
// exp(-sum(density * dt)) accumulated through the cloud march. 1.0
// means "no cloud blocking" (no clouds at all, ray misses the layer,
// or clouds_p1.w == 0); 0.0 means "fully opaque cloud." Sky-behind-
// cloud illumination of the cloud (in-scatter glow from moon / sun /
// skylight) is handled by the EXISTING b>0 / cloud-march integration
// in the path tracer and is unaffected by this G-buffer write -- this
// texture only carries the camera-ray transmittance number that the
// composite needs.
//
// Allocated whenever denoiser_active_ (alongside denoise_color,
// depth_tex, motion_tex). When the denoiser is off, the path tracer's
// inline swapchain write handles celestial attenuation naturally via
// the existing volumetric integration, so this texture is unbound
// and write is gated on denoiser_enabled. Same SPIR-V / MSL
// declaration-on-both-targets rule as albedo_tex (see that field's
// comment) -- excluding the declaration from one backend would shift
// every subsequent texture slot.
[[vk::binding(22, 0), vk::image_format("r32f")]] RWTexture2D<float>  cloud_trans_tex;

// MetalFX specular-guidance G-buffers (issue #118). Apple's
// MTLFXTemporalDenoisedScaler exposes three optional specular inputs --
// without them the scaler can't differentiate specular from diffuse
// response and exhibits 8x8 halos around bright reflections / metals.
// PR #114 wired normal + diffuse_albedo; this PR fills the remaining
// trio so MetalFX can close the gap with DLSS Ray Reconstruction on
// Apple Silicon.
//
// Binding numbers: 24/25/26. Binding 23 is reserved for #115 SIGMA
// shadow visibility (parallel-agent coordination). The cluster
// 22 (cloud_trans) / 23 (shadow_vis, #115) / 24-26 (this PR) lives
// just past the SDF-cluster buffer at 21 and keeps the descriptor
// pool sizing arithmetic simple.
//
// All three are declared on BOTH SPIR-V and Metal targets so Slang's
// declaration-order MSL slot assignment stays stable -- same rule as
// normal_tex / albedo_tex / cloud_trans_tex (see albedo_tex's comment
// for the PR #108 history that motivates this). Runtime writes are
// gated on the matching push.write_specular_*_gbuffer flag the engine
// sets only for MetalFX-family denoiser kinds.

// specular_albedo: per-pixel F0 (Fresnel reflectance at normal
// incidence). RGBA16F so it can carry the full RGB metal tint
// (metal F0 = surface albedo, NOT a uniform value -- gold is yellow
// because its specular reflectance is yellow, not because it tints
// a white diffuse). Dielectrics use the standard float3(0.04) IOR=1.5
// Schlick approximation; pure Lambert writes float3(0) since it has
// no specular lobe. Sky / no-hit pixels write float3(0).
//
// Declared as WTexture2D (write-only) rather than RWTexture2D because
// Apple Silicon GPUs cap compute kernels at 8 read_write textures per
// dispatch, and PathTrace.slang is already at that ceiling (output,
// accum_hdr, denoise_color, depth_tex, motion_tex, normal_tex,
// albedo_tex, cloud_trans_tex). Slang's WTexture2D lowers to
// `texture2d<T, access::write>` on Metal -- not counted against the
// 8-RW cap. Vulkan's GLSL/SPIR-V emit uses a write-only `imageStore`
// path; the host-side bind path is identical (storage image, same
// VkDescriptorSet entry, same vk::binding number). Issue #118 / Slang
// docs WTexture2D guidance: a2-03-wgsl-target-specific.md.
[[vk::binding(24, 0), vk::image_format("rgba16f")]] WTexture2D<float4> specular_albedo_tex;

// roughness: linear surface roughness in [0, 1]. Single-channel.
// 0 = perfect mirror, 1 = fully Lambertian. The path tracer's
// material struct already carries a roughness param on PBR-style
// hits (used in the GGX BRDF); this G-buffer just exposes it to
// MetalFX as a guidance input so the scaler can vary its spatial
// filter aggressiveness by surface response. Sky / no-hit pixels
// write 1.0 (fully rough -- MetalFX won't try to sharpen a specular
// highlight in empty space). See specular_albedo_tex's comment for
// the WTexture2D vs RWTexture2D rationale (Apple's 8-RW cap).
[[vk::binding(25, 0), vk::image_format("r32f")]] WTexture2D<float>  roughness_tex;

// specular_hit_distance: distance from the camera (in world units) to
// the surface a reflection ray from the primary hit would land on.
// Single-channel. Apple uses this for parallax-correct temporal
// reprojection of specular highlights -- without it, reflections on
// moving cameras smear because MetalFX has no way to know the
// reflected-virtual-image depth differs from the visible-surface
// depth.
//
// MVP semantics: we don't run a second trace per pixel (the full
// design would re-trace in r = reflect(view_dir, normal) from the
// primary hit). Instead we use `primary_t` (the distance the camera
// ray itself travelled) as a proxy, biased by the roughness: at
// roughness=0 (perfect mirror) the reflected hit is at infinity for
// curved surfaces / at primary_t for flat ones, so this proxy is
// closer to right for shorter-distance reflections; at roughness=1
// the field doesn't matter anyway (MetalFX won't try to reproject
// a fully-rough surface as a sharp specular). A future PR can
// replace this with a real reflection-ray trace; the visual diff
// determines whether the perf cost is worth it.
// Sky / no-hit pixels write 0 (no surface). See specular_albedo_tex's
// comment for the WTexture2D vs RWTexture2D rationale.
[[vk::binding(26, 0), vk::image_format("r32f")]] WTexture2D<float>  specular_hit_distance_tex;

// --- Wave 8 ocean (#25) ----------------------------------------------------
// FFT Tessendorf ocean displacement + normal textures, written CPU-side by
// pt::ocean::OceanFFT and uploaded each frame (RGBA32F). The path tracer
// SAMPLES them (read-only) while ray-marching the displaced heightfield off
// any MAT_WATER plane, so -- exactly like env_map / star_map / moon_map --
// they're declared as read-only Texture2D on Metal (which keeps them OFF
// Apple Silicon's 8-RW-texture compute cap that PathTrace already saturates)
// and RWTexture2D on Vulkan (whose descriptor layout already wires storage
// images; switching to sampled images there would cascade through the
// pipeline layout for no gain). vk::binding 32/33 sit one past the SPH
// particle SSBO's 31 -- the next free bindings on the integration branch.
//
// ocean_displacement: .xyz = world-space displacement (m, y = height,
//                     xz = choppy Gerstner lateral push), .w = foam in
//                     [0,1]. ocean_normal: .xyz = unit world normal.
// Both tile every ocean_params0.y metres of world XZ. Gated entirely by
// ocean_params0.x -- when 0 the MAT_WATER branch never touches these.
#ifdef PT_TARGET_METAL
Texture2D<float4> ocean_displacement;
Texture2D<float4> ocean_normal;
#else
[[vk::binding(32, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> ocean_displacement;
[[vk::binding(33, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> ocean_normal;
#endif
// --- end Wave 8 ocean ------------------------------------------------------
// --- Wave 8 PBR (#26) -- material texture atlas ---------------------------
// All PBR material maps (albedo / normal / roughness / metallic) for every
// material are packed into ONE vertical-strip atlas so the path tracer
// takes exactly ONE extra texture binding. Layout: kPbrTileSize-square
// tiles stacked vertically (atlas is kPbrTileSize wide, kPbrAtlasTiles *
// kPbrTileSize tall). A material's per-map tile index selects the vertical
// band; samplePbrTileLinear() wraps the in-tile UV with frac() so tiling
// textures repeat cleanly.
//
// FORMAT: RGBA8_UNORM linear storage. The engine stores source PNG/JPG
// bytes (which for albedo are sRGB-encoded) AS-IS, and the shader applies
// the sRGB EOTF only to albedo samples via samplePbrTileSrgb() -- so we
// never double-gamma, and the same atlas hosts both colour (albedo) and
// linear data (normal / roughness / metallic) maps. The shader reads
// float4 (the RHI normalizes RGBA8 -> [0,1] on fetch).
//
// CRITICAL -- declared LAST in texture-declaration order (after ocean_normal,
// which Wave 8 #25 placed at MSL index 15) so on Metal pbr_atlas lands at
// MSL texture index 16 WITHOUT shifting any of the existing 0..15 texture-
// index assignments. On Metal it's a sample-only Texture2D (no vk::binding)
// -- this escapes Apple Silicon's 8-read_write-texture compute cap exactly
// like env_map / star_map / moon_map / ocean_* do, and the textured path is
// verified correct there. On Vulkan it's an RWTexture2D storage-image view
// at vk::binding 34 -- ocean (#25) took 32/33 on the integration branch, so
// PBR rebased one slot up to 34 (pbr_atlas) + 35 (mesh_uvs). The kernel only
// ever READS it (uploaded once by the engine via WriteTexture).
//
// KNOWN LIMITATION (Vulkan/MoltenVK): the textured render is correct on
// Metal + the flat-color path is bit-perfect on all three backends, but
// sampling this atlas returns zero on MoltenVK (textured surfaces read
// black). This is the "Mac-Vulkan pixel-correctness historically
// unverified" class of issue -- a high storage-image descriptor binding on
// Apple Silicon. RGBA32F was tried (env_map's working format) with no
// change, so it is a descriptor-wiring quirk, not a format issue. Deferred:
// native-Vulkan (NVIDIA) verification + a SAMPLED_IMAGE+sampler descriptor
// path are the likely fixes.
#ifdef PT_TARGET_METAL
Texture2D<float4> pbr_atlas;
#else
[[vk::binding(34, 0), vk::image_format("rgba8")]] RWTexture2D<float4> pbr_atlas;
#endif
// Tile geometry -- MUST match Engine.h's kPbrTileSize / kPbrAtlasTiles.
static const uint kPbrTileSize   = 256u;
static const uint kPbrAtlasTiles = 16u;
static const uint kPbrNoTexTile  = 0xFFFFFFFFu;
// Mip pyramid (fixes horizon aliasing when a tile minifies into the far
// field). Every tile band now carries a full box-filtered mip chain:
//   level 0     : the base kPbrTileSize-square image at x=[0, kPbrTileSize)
//   levels 1..8 : packed into the right column x=[kPbrTileSize, 3/2*kPbrTileSize)
//                 stacked vertically inside the SAME 256-row band
//                 (128+64+...+1 = 255 rows <= kPbrTileSize).
// The atlas is therefore kPbrTileSize*3/2 wide (was kPbrTileSize); the band
// height (and kPbrAtlasTiles*kPbrTileSize atlas height) is unchanged. LOD is
// picked per hit from a ray cone (no ddx/ddy on analytic prims). Engine.h's
// kPbrAtlasWidth / kPbrMaxLod build + upload the pyramid; these MUST match.
static const uint kPbrAtlasWidth = kPbrTileSize + kPbrTileSize / 2u;  // 384
static const uint kPbrMaxLod     = 8u;   // log2(kPbrTileSize=256)
// --- end Wave 8 PBR -------------------------------------------------------

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

// --- SDF Phase 1 (#97) -----------------------------------------------------
// SDF cluster buffer. Each cluster is a flat array of nodes (op-tree)
// plus a tight world-space AABB. Sphere-tracing is bounded to the AABB
// after a separate (per-cluster) linear hit test that runs alongside
// the analytic-prim BVH walk. The buffer is packed in the host as a
// SoA stream of float4s; the loader below reassembles the cluster
// header + nodes.
//
// IMPORTANT: declared AFTER tri_bvh_* so existing Metal slots 0..9 are
// undisturbed (Metal slot allocation is declaration-order based; see
// the note above bvh_nodes). SDF cluster buffer lands on Metal slot 10.
// On Vulkan the explicit vk::binding(21) is what matters -- moved from
// the original binding 19 to make room for PR #106's triangle BVH at
// bindings 19/20.
[[vk::binding(21, 0)]] StructuredBuffer<float4> sdf_clusters;
// --- end SDF Phase 1 -------------------------------------------------------

// Sun-NEE shadow + direct-sun G-buffer (issue #115, B10 demod). FOUR
// floats per pixel, row-major with a stride of 4 (base = 4*(y*width+x)):
//   [+0] sun visibility -- 1.0 = primary hit fully lit; 0.0 = fully in
//        shadow; partial from the spp average. Sky pixels (no primary
//        hit) write 1.0 so SigmaShadow never darkens the sky composite.
//   [+1..3] S.rgb -- the UNSHADOWED primary-hit direct-sun radiance
//        (throughput * sun_rad * brdf * n_dot_l, aerial-perspective and
//        spectral weighted to match the sun term inside frame_radiance).
//        SigmaShadow re-modulates ONLY this term post-denoise
//        (hdr += S * vis_denoised) after PathTrace subtracted the
//        shadowed sun from the denoiser input -- so the sun shadow stays
//        crisp while lights / GI in the umbra are untouched.
// Non-Lambert primary hits, sky pixels, and sun-below-horizon samples
// leave S = 0, so the re-modulation is a no-op there.
//
// Storage buffer (NOT texture) is deliberate. Metal's compute pipeline
// caps RWTexture2D bindings at 8 read_write textures per kernel.
// PathTrace already sits exactly at that limit: output, accum_hdr,
// denoise_color, depth_tex, motion_tex, normal_tex, albedo_tex,
// cloud_trans_tex. A ninth RW texture would fail pipeline
// compilation on Apple Silicon. Storage buffers (RWStructuredBuffer)
// don't count toward that quota -- same workaround DenoiseAtrous.slang
// uses for its variance + moments buffers. Indexing is row-major
// `y*width + x` so the downstream SigmaShadow kernel can read
// neighbours via the same arithmetic.
//
// IMPORTANT: declared AFTER sdf_clusters so existing Metal slots
// 0..10 stay undisturbed. Slang's MSL backend assigns Metal buffer
// slots in DECLARATION order (AS first, then storage buffers); the
// vk::binding(N) is the Vulkan-only descriptor index. Inserting this
// buffer anywhere earlier in the file would shift exposure_state /
// bvh_nodes / tri_bvh_* / sdf_clusters up by one MSL slot and break
// every engine BindBuffer call past the insertion point (the
// drilled-down lesson behind the comment above sdf_clusters about
// declaration order being load-bearing). Lands on Metal slot 11,
// matching the engine's BindBuffer(11, ...) call site.
//
// Allocated alongside the other denoiser textures whenever
// denoiser_active_ AND r_shadow_demod != 0. The runtime
// `write_shadow_vis` push flag is the actual run-or-skip switch; when
// 0 the write site below elides (saves a 4-byte write per pixel per
// frame on the legacy-shadow path). PathTrace declares the binding
// unconditionally so Slang's MSL backend doesn't dead-code-eliminate
// the slot and shift subsequent buffer indices -- same lesson the
// albedo_tex declaration captures upstream.
[[vk::binding(23, 0)]] RWStructuredBuffer<float> shadow_vis_buf;

// --- Light primitives (#73) ------------------------------------------------
// Analytic light list for direct-lighting NEE: point, spot, sphere
// (area), and quad (area) lights. CPU rebuilds this buffer when the
// `light_*` console commands mutate the engine's light_prims_ map; the
// shader's Lambert NEE block picks one entry per sample (uniform over
// the count) and traces a shadow ray to a sampled point on it.
//
// Layout: 5 float4s per light (matches the analytic primitive pattern's
// declaration-order convention, sized to fit every supported type).
//
//   v0.xyz = position             (point/spot center, sphere center,
//                                   quad center)
//   v0.w   = radius_or_param      (sphere radius in metres; point/spot/
//                                   quad: unused -> 0)
//   v1.rgb = intensity            (per-channel radiant intensity in
//                                   W/sr for point/spot; per-area
//                                   radiance W/m^2/sr for sphere/quad;
//                                   shader integrates by sampled area
//                                   so the user can read these as the
//                                   "physical" emitter brightness)
//   v1.w   = type_as_float        (LIGHT_POINT=0, LIGHT_SPOT=1,
//                                   LIGHT_SPHERE=2, LIGHT_QUAD=3)
//   v2.xyz = direction            (spot axis, unit vec; quad normal,
//                                   unit vec; point/sphere: unused)
//   v2.w   = cos_outer            (spot outer half-angle cosine; spot
//                                   only)
//   v3.xyz = u_extent_or_inner    (quad u-axis half-extent vector, or
//                                   for spot: x = cos_inner)
//   v3.w   = v_half_extent        (quad v-axis half-extent magnitude;
//                                   the v-axis is implicit:
//                                   normalize(cross(normal, u)))
//   v4.xyzw = orient quaternion   (rotation gizmo follow-up to #206;
//                                   xyzw, identity (0,0,0,1) leaves the
//                                   stored dir / u_vec unchanged. Spot
//                                   rotates `dir` before the cone test;
//                                   quad rotates `dir` and `u_vec`
//                                   before area-sample math.)
//
// IMPORTANT: declared AFTER shadow_vis_buf so existing Metal slots 0..11
// are undisturbed (Metal slot allocation is declaration-order based).
// Light buffer lands on Metal slot 12. On Vulkan the explicit
// vk::binding(27) is what matters; sits past the SIGMA #115 / MetalFX
// specular #118 reservations at 23..26.
[[vk::binding(27, 0)]] StructuredBuffer<float4> light_prims;
// --- end Light primitives --------------------------------------------------

// --- Light tree (#129) -----------------------------------------------------
// Hierarchical light tree (Conty Estevez & Kulla 2018) packed node array.
// 4 float4s per node (LightTreeNode in src/renderer/LightTree.h).
// Declared AFTER light_prims so Metal slot allocation lands at slot 13;
// Vulkan binding 28.
//
// PT_LIGHT_TREE_ENABLED (CMake option PT_LIGHT_TREE, default ON) gates
// the consuming code (functions + the picker if-branch). The binding
// itself is *always declared* so Metal buffer-slot assignment stays
// stable across PT_LIGHT_TREE=ON / =OFF builds. When the gate is OFF,
// the load helpers + traversal are compiled out (smaller megakernel,
// less ICache+register pressure on Apple Silicon -- see
// `perf/megakernel-split-light-tree` regression fix). The host-side
// Engine continues to upload the SSBO either way; the unused binding
// in the disabled build is benign (Slang strips its parameter from
// the emitted shader, but the descriptor-set layout on Vulkan is
// PARTIALLY_BOUND and the Metal slot table tolerates an unused index).
[[vk::binding(28, 0)]] StructuredBuffer<float4> light_tree_nodes;

// --- ReSTIR DI Phase A reservoir buffer (#78) ------------------------------
// Per-pixel WRS reservoir SSBO. Declared AFTER light_tree_nodes:
// Metal slot 14, Vulkan binding 29.
[[vk::binding(29, 0)]] RWStructuredBuffer<float4> reservoir_curr_buf;

// --- Fluid Phase 1 (#136) -- smoke emitters --------------------------------
// Density-injection plume list consumed by cloud_density_at() in
// PathTraceCloud.slang. 3 float4s per emitter (48 B):
//   v0.xyz = base position (world space, metres)
//   v0.w   = Gaussian falloff radius (metres)
//   v1.xyz = parametric drift velocity (m/s)
//   v1.w   = peak additive sigma_t (per metre)
//   v2.xyz = colour tint (multiplier on the smoke fraction's in-
//            scattered radiance at the cloud-march sample point;
//            (1,1,1) = neutral white smoke). Read by
//            smoke_density_add_tinted() and density-weighted
//            averaged across overlapping plumes before being
//            multiplied into the per-light NEE contributions.
//            The scalar-only shadow paths (cloud_optical_depth /
//            cloud_optical_depth_stratified) intentionally ignore
//            this: tint modulates emission TOWARD the camera, not
//            the transmittance imposed on the sun-NEE leg.
//   v2.w   = falloff exponent on (r/radius); 2 = Gaussian-ish
//
// Engine bounds the iteration count to push.smoke_params.x (host clamps
// to kMaxSmokeEmitters=16). Phase 1: parametric drift only, no advection.
// Phase 2 (#180): each emitter expands into a chain of `puff_count`
// time-staggered Gaussians; per-puff age / fade / advection / radius
// expansion live entirely in the shader, driven by push.smoke_params2
// -- no per-emitter storage changes were needed.
//
// Declared AFTER reservoir_curr_buf so Metal slot lands at 15; the
// shader's matching vk::binding is 30 (one past ReSTIR's 29). The agent
// originally targeted binding 28 / slot 13 but that's owned by light
// tree #129 -- moved during integration merge to avoid the collision.
[[vk::binding(30, 0)]] StructuredBuffer<float4> smoke_emitters;
// --- end Fluid Phase 1 -----------------------------------------------------

// --- Fluid Phase 3 (#22) -- SPH particle splat list ------------------------
// SPH particle list packed 2 float4s per particle (32 B):
//   v0.xyz = world-space position (m)
//   v0.w   = splat radius (m)
//   v1.x   = splat density (peak additive sigma_t; per-particle Gaussian
//            magnitude in the same units as smoke_emitters' v1.w)
//   v1.y   = age normalised [0..1] (0 = freshly emitted, 1 = end of life)
//   v1.zw  = reserved
//
// The host runs a CPU SPH solver (pt::sph::SmokeSPH) that owns the
// particle pool + spatial hash + neighbour-summed pressure/viscosity
// integration; the GPU only sees the integrated splat list each frame.
// Cap is push.smoke_sph_params.x particles -- the engine clamps it on
// upload and the shader bounds the loop with `[loop]` so the SPV /
// MSL backend produces a tight (variable-bound) loop rather than
// unrolling.
//
// Engine slot 16 / vk::binding 31 lands one past smoke_emitters
// (slot 15 / binding 30). Always-bound (placeholder fallback) for the
// same max-bound + 1 push-slot rule that smoke_emitters obeys; the
// shader's smoke_sph_density_add gates on push.smoke_sph_params.x > 0.
[[vk::binding(31, 0)]] StructuredBuffer<float4> sph_particles;
// --- end Fluid Phase 3 -----------------------------------------------------

// --- Wave 8 PBR (#26) -- mesh UVs -----------------------------------------
// Per-vertex texture coordinates parallel to mesh_positions (2 floats =
// u, v per vertex). float-typed (not float2) for the same reason
// mesh_positions is float-typed: Slang gives StructuredBuffer<float2> a
// 16-byte stride on MSL while the host uploads tightly-packed 8-byte
// float2. The shader reads mesh_uvs[vid*2 + 0/1].
//
// CRITICAL -- declared LAST in buffer-declaration order (after
// sph_particles) so on Metal it lands at MSL buffer slot 17 (one past
// sph_particles' slot 16) WITHOUT shifting any of the existing 1..16
// buffer-slot assignments. The engine binds it at engine slot 17 and the
// push-constant slot moves to 18 (max-bound + 1) accordingly. (Ocean #25's
// displacement/normal are TEXTURES, not buffers, so they don't shift the
// Metal buffer-slot numbering -- mesh_uvs stays at buffer slot 17.) On
// Vulkan the explicit vk::binding(35) is what matters: ocean (#25) took
// 32/33 and PBR's pbr_atlas rebased to 34, so mesh_uvs sits at 35 (the
// Frame UBO claims binding 14, env CDFs 10/11). Only read on a mesh hit
// when push.mesh_tex_indices has a non-kPbrNoTexTile channel; when the
// mesh has no UVs the host binds a placeholder and forces the tile indices
// to kPbrNoTexTile so the buffer is never sampled.
[[vk::binding(35, 0)]] StructuredBuffer<float>  mesh_uvs;
// --- end Wave 8 PBR -------------------------------------------------------

// --- Multiple scattering (#280) --------------------------------------------
//
// Hillaire 2020 section 4's Psi_ms, 32 x 32 over (radius, sun-zenith
// cosine), one float4 per texel: rgb is the isotropic second-and-higher-
// order in-scattered radiance WITH the solar irradiance already folded in,
// w is 1.0 on every texel the host wrote.  A zero-filled placeholder
// therefore reads w == 0, which is what `ptMsLutReady` below tests -- a
// buffer that was never built contributes nothing rather than silently
// contributing a table of zeroes that looks like "multiple scattering is
// negligible".
//
// BUILT ON THE HOST (src/renderer/Atmosphere.cpp), not by a compute pass.
// Declared right after mesh_uvs (binding 35 / MSL slot 17) so on Metal it
// lands at MSL slot 18; the host binds it there and the push block moves
// one past it, which MetalPipeline reads out of the emitted MSL rather than
// assuming. On Vulkan the explicit vk::binding(46) is what matters.
[[vk::binding(46, 0)]] StructuredBuffer<float4> atmo_ms_lut;

bool ptMsLutReady() { return atmo_ms_lut[0].w > 0.5; }
// The bilinear fetch itself is `ptAtmoMultiScatter`, next to
// atmosphereBody() further down -- it needs PtAtmoBody, which arrives with
// the PathTraceMath import below this declaration block.
// --- end multiple scattering ------------------------------------------------

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

// The full PtPush is 832 bytes today (112 B head + 720 B tail; see the
// static_assert at the dispatch site in Engine.cpp -- grows whenever a
// new vec4 / uvec4 field is appended). Native Vulkan / NVIDIA caps push
// constants at 256B, so on SPIR-V we keep only the small per-dispatch
// prefix (camera basis + flags + jitter, 112B) in [[vk::push_constant]]
// and spill the rest into a UBO at binding(14, 0).
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
    //   .w = blade rotation in radians (Wave 10; orients the polygon,
    //        ignored for the round disk).
    float4 dof_params;
    // --- Wave 10 bloom/bokeh ---
    // Polygonal-bokeh extras.
    //   .x = cat's-eye / optical-vignetting strength (0 = round bokeh
    //        across the frame, bit-for-bit with the pre-wave-10 aperture;
    //        1 = strong off-axis lemon clipping).
    //   .y/.z/.w reserved.
    float4 dof_bokeh_params;
    // --- end Wave 10 bloom/bokeh ---
    // --- Wave 8 spectral (#27) ---
    // Spectral dispersion (hero-wavelength) parameters.
    //   .x = enabled flag (0 = achromatic RGB dielectric path, bit-stable
    //        with pre-#27 goldens; 1 = per-ray hero-wavelength sampling
    //        with wavelength-dependent IOR -> real prism dispersion).
    //   .y = default Cauchy C coefficient in micrometre^2 for dielectric
    //        materials (crown glass BK7 ~= 0.00420). The material's nominal
    //        IOR (h.ior) anchors the Cauchy curve at the d-line (587.6 nm).
    //   .z/.w reserved.
    float4 spectral_params;
    // --- end ---
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
    //   clouds_p4: (curl_amount, curl_scale, erosion_amount, _reserved)
    //              issue #117. curl_amount in metres is the magnitude
    //              of the Bridson-2007 curl-noise displacement applied
    //              to the noise sample position before density eval --
    //              gives the cloud body sheared filamentous detail
    //              (cumulus edge wisps, cirrus streamers). 0 = bypass
    //              the curl displacement entirely (regression-safe).
    //              curl_scale is the curl-noise frequency in cycles
    //              per metre (independent of cp2.z so coarse cloud
    //              bodies can carry fine curl turbulence).
    //              erosion_amount is a secondary high-frequency edge
    //              erosion term added on top of cp3.z; eats soft
    //              cumulus margins into wisps. 0 = bypass.
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 clouds_p4;
    // Moon. .xyz = unit vector toward the moon (engine computes via
    // Meeus simplified moon-position formulas + observer lat/lon).
    // .w = phase angle in radians (0 = new, π = full); used for both
    // the moon disc's brightness in skyColor and for moonlight NEE
    // contribution amplitude on cloud / surface BRDF rays.
    float4 moon_dir_phase;
    // Moon extras. .x = r_moon_size, .y = astronomical distance ratio
    // (mean/current; perigee ~1.06, apogee ~0.95), .z = r_moon_disc_brightness
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
    // from 1.0 above ~30deg.
    // .w = planet radius in metres for spherical-Earth atmospheric
    // transmittance (issue #51). When > 0, `atmosphericTransmittance`
    // numerically integrates Mie + Rayleigh optical depth along the
    // curved chord through a thin shell around a sphere of this radius
    // (centre at world origin offset by -r so the y=0 plane sits on
    // the surface). 0 falls back to the legacy planar exponential
    // integral. WGS-84 Earth is 6,378,137 m.
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
    // --- SDF Phase 1 (#97) + Phase 3 (#99) ---------------------------------
    // SDF traversal params. See Engine.cpp's PtPush::sdf_params for full
    // docs.
    //   .x = sdf_cluster_count -- number of SDF clusters in
    //        `sdf_clusters`. 0 disables the SDF path entirely.
    //   .y = sdf_max_iters     -- sphere-trace STEP cap per cluster
    //        (host cvar r_sdf_max_iters, default 128, clamped to
    //        1..256 on the host). Applies to both analytic SDF and
    //        fractal clusters.
    //   .z = sdf_debug_iters   -- 1 -> output an iteration heat-map
    //        instead of normal shading for diagnostics. Phase 1 leaves
    //        the heat-map plumbed but not yet visualized; the field is
    //        wired so Phase 2 / fractal tuning can switch it on without
    //        a push-constant reshuffle.
    //   .w = sdf_fractal_iters -- Phase 3, per-DE bounded iteration
    //        count for fractal SDFs. Independent of .y. Clamped 1..32
    //        on host. (Phase 2's sdf_normal_mode toggle lives at
    //        sdf_params_f.w because Phase 3 claimed this slot first.)
    uint4 sdf_params;
    // .x = r_sdf_epsilon (sphere-trace surface-distance terminator,
    //      metres). Default 1e-4.
    // .y = r_sdf_fractal_power (Phase 3, default Mandelbulb power 8).
    // .z = r_sdf_de_eps_scale (Phase 3, iteration-relaxed eps scale,
    //      default 1.0).
    // .w = r_sdf_normal_mode (Phase 2 #98; encoded as float because
    //      Phase 3 claimed sdf_params.w first). 0.0 = forward-mode AD,
    //      1.0 = 6-tap central differences.
    float4 sdf_params_f;
    // --- end SDF Phase 1 + Phase 3 -----------------------------------------
    // Celestials composite gate (issue #46). 1 = subtract
    // starsOnly(rd) + sunDisc(rd) + moonDisc(rd) from the b==0
    // primary-miss sky term so denoise_color reaches SVGF/MetalFX
    // celestial-free; the engine's StarsComposite kernel re-adds them
    // post-denoise with N aperture-sampled rays per pixel (pixel-
    // sharp stars, clean single-frame bokeh on sun/moon). 0 = legacy
    // behaviour, celestials fold into frame_radiance and go through
    // the denoiser. Engine fills this from r_star_split gated by
    // denoiser_enabled AND procedural sky mode. The trailing 12 bytes
    // of pad keep the cbuffer trailing block 16-byte aligned for
    // std140 / MSL. Subtraction is gated on b == 0 only -- secondary
    // bounces (b > 0) keep the full skyColor() so stars still
    // contribute to indirect illumination of the scene.
    uint composite_celestials;
    uint _pad_celestials0;
    // SIGMA shadow demod gate (issue #115). 1 -> write per-primary-hit
    // sun-NEE shadow-ray transmittance into `shadow_vis_buf` at the
    // end of main(); the engine's SigmaShadow kernel denoises that
    // buffer with a 5x5 bilateral and multiplies the denoised visibility
    // into the post-radiance-denoise HDR.
    uint write_shadow_vis;
    uint _pad_shadow_vis;
    // --- Heterogeneous volumetric phase 4 (#100) ---------------------------
    // Three floats sharing the trailing block. Host pushes them from
    // r_vol_density_scale / r_vol_phase_g / r_vol_multiscatter_bounces
    // (engine-side CVar names; the shader names below are kept
    // descriptive: vol_phase_g_cloud disambiguates from the atmospheric-
    // haze g, and vol_multiscatter_oct is the shader-side octave count
    // even though the user-facing CVar is named for the "bounces"
    // abstraction).
    //
    //   vol_density_scale     -- linear multiplier on cloud peak sigma_t
    //                            so artists can dial thickness up/down.
    //   vol_phase_g_cloud     -- HG anisotropy for the cloud march
    //                            (decoupled from vol_params.y haze g).
    //                            +0.8 = canonical cumulus forward lobe.
    //   vol_multiscatter_oct  -- Wrenninge-style multi-scatter octave
    //                            count (clamped 0..4 in shader). 0 =
    //                            pure single-scatter; >=1 fakes
    //                            diffuse fill so cloud cores don't
    //                            read pitch-black under sun shadow.
    //                            Default 2.
    float vol_density_scale;
    float vol_phase_g_cloud;
    float vol_multiscatter_oct;
    float _pad_vol;
    // --- Light primitives (#73) + Light tree (#129) ------------------------
    // light_count: number of analytic light primitives in `light_prims`.
    //              0 disables the Lambert NEE-to-lights branch entirely.
    // light_tree_node_count: number of nodes in `light_tree_nodes`. 0
    //              means the host didn't build a tree this frame.
    // light_tree_enabled: 1 -> use O(log N) tree traversal for light
    //              selection; 0 -> fall back to #73's uniform single
    //              pick. Set by the engine from the r_light_tree cvar
    //              (when 0, the cvar acts as a runtime kill switch).
    // _pad_light_tail: trailing pad to keep the 16-byte block aligned.
    uint light_count;
    uint light_tree_node_count;
    uint light_tree_enabled;
    uint _pad_light_tail;
    // MetalFX specular-guidance G-buffer write gates (issue #118).
    // The engine sets these only for MetalFX-family denoiser kinds
    // (DenoiserKind::MetalFX / SvgfBasicMetalFx / SvgfAtrousMetalFx)
    // and only when the matching G-buffer texture is actually
    // allocated this frame -- same pattern as write_normal_gbuffer /
    // write_albedo_gbuffer. Under PARTIALLY_BOUND descriptors the
    // shader must elide the write when the texture is unbound, so
    // each one gets its own runtime gate rather than collapsing into
    // a single MetalFX flag (lets future modes pick + choose; e.g.
    // a roughness-only diagnostic kernel could write just the
    // roughness gbuffer). Trailing pad keeps the 16-byte alignment
    // the std140 / MSL cbuffer rule demands.
    uint write_specular_albedo_gbuffer;
    uint write_roughness_gbuffer;
    uint write_specular_hit_distance_gbuffer;
    uint _pad_specular_gbuffers0;
    // --- Water Phase 1 (#134) -------------------------------------------------
    // See Engine.cpp's PtPush::water_params0 / water_params1 for full docs.
    //   water_params0.xyz = Beer's-law absorption per channel (1/m).
    //                  .w = refractive index (clamped to [1.0, 2.4] below).
    //   water_params1.x  = wave_scale (cycles / m).
    //                  .y = wave_amplitude (normal perturbation strength).
    //                  .z = wave_speed (animation rate multiplier).
    //                  .w = time_seconds (shared with cloud-wind accumulator).
    float4 water_params0;
    float4 water_params1;
    // --- end Water Phase 1 ----------------------------------------------------
    // --- ReSTIR DI Phase A (#78) -------------------------------------------
    uint restir_enabled;
    uint restir_k_candidates;
    uint _pad_restir0;
    uint _pad_restir1;
    // --- end ReSTIR DI Phase A -----------------------------------------------
    // --- Fluid Phase 1 (#136) -- smoke emitters -------------------------------
    // smoke_params.x = smoke_count (clamped to kMaxSmokeEmitters=16)
    //             .y = smoke_enabled (0/1; r_smoke_enabled gate)
    //             .z = time_seconds for parametric drift
    //             .w = r_cloud_attenuate_background flag (57fd9ff)
    float4 smoke_params;
    // --- end Fluid Phase 1 ----------------------------------------------------
    // --- Fluid Phase 2 (#180) -- advected emission ----------------------------
    // smoke_params2.x = r_smoke_lifetime          (seconds, >= 0)
    //              .y = r_smoke_advection_strength (multiplier on emitter vel)
    //              .z = r_smoke_puff_count        (1..16, integer-as-float)
    //              .w = r_smoke_expansion_rate    (per-second radius growth)
    float4 smoke_params2;
    // --- end Fluid Phase 2 ----------------------------------------------------
    // --- Fluid wave-6 -- "looks like smoke" pass ------------------------------
    // smoke_params3.x = r_smoke_turbulence_scale   (cycles per puff radius;
    //                    larger = finer wisps, ~4 = high-freq fine detail).
    //              .y = r_smoke_turbulence_amplitude (0=no modulation; 1=full
    //                    replacement of the Gaussian by noise-modulated
    //                    density; ~0.6 reads as natural).
    //              .z = r_smoke_decay_shape (exponent on the age-fade curve.
    //                    1 = legacy linear (1 - age/lifetime); 2 = quadratic
    //                    so heads stay denser and tails wisp away faster.
    //                    3-4 give a sharper "fresh head, vanishing tail" look).
    //              .w = r_smoke_age_tint_strength (0..1 amount of age-driven
    //                    warm->cool tint applied to fresh vs old puffs. 0
    //                    disables the gradient; 0.2-0.3 reads as natural
    //                    real-world smoke; 1.0 maxes out the warm/cool
    //                    contrast across the lifetime).
    float4 smoke_params3;
    // --- end Fluid wave-6 -----------------------------------------------------
    // --- Fluid Phase 3 (#22) -- SPH fluid sim ---------------------------------
    // smoke_sph_params.x = sph_particle_count   (0 disables SPH consumption;
    //                       host clamps to push.smoke_sph_params.w cap)
    //                 .y = smoke_mode (0 = procedural, 1 = sph, 2 = both)
    //                       The integer values are bit-encoded so the
    //                       shader gate is a single int cast; only mode 1
    //                       and mode 2 read sph_particles[].
    //                 .z = render_density_floor (clamp value subtracted
    //                       from the per-particle splat density so an
    //                       in-cell quiescent particle reads as zero --
    //                       prevents the SPH-mode rest-density from
    //                       filling the world with a uniform fog).
    //                 .w = particle_buffer_cap (host-side max upload,
    //                       informational; the shader only iterates
    //                       smoke_sph_params.x particles).
    float4 smoke_sph_params;
    // --- end Fluid Phase 3 ----------------------------------------------------
    // --- Motion blur (#85) ----------------------------------------------------
    // motion_blur_params.x = r_motion_blur (1 = stochastic shutter-time
    //                         sampling for per-ray motion blur; 0 = legacy
    //                         single-frame sphere positions, bit-exact vs
    //                         pre-#85 even though the primitive buffer
    //                         still carries v3=prev_pos for every prim).
    //                .y = r_shutter_speed in seconds. Currently informational
    //                      only -- the shader's lerp factor is U(0,1) over
    //                      the prev/curr window the host snapshots once
    //                      per StepPhysics tick; a longer shutter just
    //                      means the user is content with the streak
    //                      magnitude produced at the current frame rate.
    //                .z/.w reserved.
    float4 motion_blur_params;
    // --- end Motion blur ------------------------------------------------------
    // --- Underwater medium tracking (Phase 4, #134) --------------------------
    // Per-ray "in-water" state Phase 4. Host scans analytic primitives for
    // MAT_WATER planes at engine push setup time and checks whether the
    // camera position sits below any of them; if so .x = 1.0 (initial
    // primary ray starts inside the water medium and gets Beer's-law
    // attenuation applied per-segment in the bounce loop). .yzw carry the
    // configurable "deep water void" RGB tint the sky-on-miss fallback
    // returns when an underwater bounce ray escapes to the empty horizon
    // (vs. returning sky-blue, which would break the deep-water illusion).
    // Defaults near-black so unconfigured scenes still read as proper
    // underwater dark on TIR / deep miss. Pulled from r_water_deep_color_r/g/b
    // cvars (registered alongside the existing r_water_* family).
    float4 water_state;
    // --- end Underwater medium tracking --------------------------------------
    // --- Editor backend (agent-19) -------------------------------------------
    // editor_params.x = selected_prim_id (0 = no selection / paint nothing).
    //              .y = r_editor_outline gate (0/1; 0 forces the outline
    //                                          off regardless of .x).
    //              .z = selected_kind  (mirrors host SelectionKind enum;
    //                                   only AnalyticPrim paints today).
    //              .w reserved.
    // Host writes the EFFECTIVE id into .x (zero when the cvar gate is
    // off or kind != AnalyticPrim) so the shader can collapse the
    // outline test to a single "if (.x != 0 && prim_id == .x)" without
    // re-reading .y / .z on every prim hit.
    uint4 editor_params;
    // --- end Editor backend ---------------------------------------------------
    // --- Clouds dual-mode gate (#24) -----------------------------------------
    // clouds_runtime.x = r_clouds_mode enum
    //                    0 = pathtraced (legacy inline cloud march runs
    //                                    in this kernel, this slot un-used)
    //                    1 = procedural_raymarched (the CloudsRaymarch
    //                                    compute pre-pass owns the cloud
    //                                    sampling; this kernel SKIPS the
    //                                    entire cloud-march block and
    //                                    also skips writing cloud_trans_tex,
    //                                    leaving the pre-pass's value
    //                                    in place for StarsComposite to read).
    //              .y/.z/.w reserved.
    // Defaults to 0 in unset configurations so the existing inline cloud
    // path stays bit-for-bit preserved for old scenes / old configs.
    uint4 clouds_runtime;
    // --- end Clouds dual-mode gate -------------------------------------------
    // --- Mesh motion blur (wave-7 #21, follow-up to motion_blur_params #85) ---
    // mesh_motion_prev.xyz = prev_translation (world-space metres). Where
    //                       the mesh was last frame. Combined with curr
    //                       below, the shader computes per-ray effective
    //                       translation lerp(prev, curr, t01) at the
    //                       shutter sample t01 ∈ [0, 1] (0 = start of
    //                       shutter, 1 = end), then shifts the mesh ray
    //                       origin by -lerp(prev, curr, t01) before
    //                       intersection. Equivalent to translating the
    //                       baked mesh by +lerp(prev, curr, t01) while
    //                       keeping the TLAS identity-transformed, so no
    //                       per-frame BLAS / TLAS rebuild is needed.
    //                 .w = mesh_motion_active flag (0/1). Host gates
    //                       this on (r_motion_blur != 0 AND prev != curr)
    //                       OR the mesh has been translated away from
    //                       its baked origin. When 0, the shader skips
    //                       the shift entirely (saves a vec3 add per
    //                       mesh hit on truly-static scenes).
    // mesh_motion_curr.xyz = curr_translation (world-space metres). Where
    //                       the mesh is this frame. When r_motion_blur
    //                       is off, the shader forces shutter_t01 = 1.0
    //                       (PR #85 plumbing) so the lerp collapses to
    //                       curr -- the mesh renders at its current
    //                       world position with zero streak.
    //                 .w = reserved.
    //
    // Layout collapses to a no-op on truly-static scenes (prev == curr ==
    // origin, .w stays 0): the shader's mesh hit path is bit-identical to
    // pre-#21. Translation preserves normals, so the existing ObjectToWorld
    // normal pipeline on the HW path is unchanged (q.CommittedObjectToWorld4x3()
    // stays identity because we keep the TLAS instance transform identity).
    float4 mesh_motion_prev;
    float4 mesh_motion_curr;
    // --- end Mesh motion blur -------------------------------------------------
    // --- Wave 8 ocean (#25) ---------------------------------------------------
    // FFT ocean ray-march parameters. See Engine.cpp's PtPush::ocean_params0
    // / ocean_params1 for the full field-by-field layout.
    //   ocean_params0.x = ocean_enabled (0 -> Phase 1 analytic wave normal)
    //                .y = tile_size_m (world XZ / tile -> displacement UV)
    //                .z = choppiness (lateral displacement scale)
    //                .w = max_disp_y (peak |height| this frame, m)
    //   ocean_params1.x = march_steps (int) ; .y = grid_size ; .zw reserved.
    float4 ocean_params0;
    float4 ocean_params1;
    // --- end Wave 8 ocean -----------------------------------------------------
    // --- Wave 8 PBR (#26) -- mesh material texture tiles ----------------------
    // .x = albedo_tex, .y = normal_tex, .z = roughness_tex, .w = metallic_tex.
    // kPbrNoTexTile (0xFFFFFFFF) on a channel means the mesh's matching map
    // is unset and the shader falls back to MESH_ALBEDO / fixed roughness.
    // Default is all-0xFFFFFFFF so an untextured mesh renders bit-identically.
    uint4 mesh_tex_indices;
    // --- end Wave 8 PBR -------------------------------------------------------
    // --- Wave 9 ocean-foam ----------------------------------------------------
    // FFT-ocean foam shading hints. The CPU solver bakes the foam coverage
    // (crest-fold + wind whitecaps + persistence) into ocean_displacement.w;
    // these only tune the BRDF blend.
    //   .x = foam_amount (intensity; 0 -> skip foam, leave the clean
    //        specular sea; >0 widens the foam-blend threshold a touch).
    //   .y = foam_coverage gain (informational; wind ramp is baked CPU-side).
    //   .zw reserved.
    float4 ocean_foam_params;
    // --- end Wave 9 ocean-foam ------------------------------------------------
    // --- Wave 9 hosek-sky ---
    // Hosek-Wilkie (2012) physically-based sky parameters. Read only when
    // sky mode == 3 (hosek). See Engine.cpp's PtPush::hosek_params.
    //   .x = turbidity (1..10; real atmospheric turbidity T).
    //   .y = ground albedo (0..1; hemispherical reflectance fed into the
    //        coefficient fit's albedo term).
    //   .z = solar zenith angle in radians (acos(sun.y), clamped so the
    //        analytic fit stays valid below the horizon).
    //   .w reserved.
    float4 hosek_params;
    // --- end Wave 9 hosek-sky ---
    // --- Wave 9 advanced materials --------------------------------------------
    // Per-prim BSDF lobe extensions, keyed by user prim id. See Engine.cpp's
    // PtPush::mat_override_count / mat_overrides for the host layout + the
    // rationale for routing these through the push constant instead of the
    // analytic-prim storage buffer (keeps kFloatsPerPrim at 28 / 7 float4s).
    //   mat_override_count.x = number of valid slots (0..kMaxMatOverrides).
    //                          0 -> the per-hit lookup is skipped entirely
    //                          so non-extended materials are bit-for-bit.
    uint4 mat_override_count;
    // kMaxMatOverrides slots, 3 float4 each. Flattened to a float4 array so
    // the std140 / MSL layout matches the host's float[96] exactly.
    //   [s*3 + 0] a: asfloat(prim_id), asfloat(ext_flags), aniso_amount,
    //                aniso_rotation
    //   [s*3 + 1] b: clearcoat_weight, clearcoat_roughness,
    //                subsurface_radius, _pad
    //   [s*3 + 2] c: subsurface_color.rgb, _pad
    float4 mat_overrides[8 * 3];
    // --- end Wave 9 advanced materials ----------------------------------------
    // --- Wave 9 tonemap (#27 follow-up) ---------------------------------------
    // tonemap_params.x = r_tonemap_op enum (0 aces / 1 agx / 2 khronos_pbr
    //                    _neutral / 3 reinhard / 4 linear). Drives both the
    //                    inline denoiser-off swapchain write here and the
    //                    matching field in Tonemap.slang's post-denoise pass.
    //                    Default 0 keeps the ACES path byte-identical so
    //                    committed goldens are unchanged.
    //              .yzw reserved (future per-operator knobs, e.g. AgX look
    //                    slope/power/sat or a Reinhard white point).
    uint4 tonemap_params;
    // --- end Wave 9 tonemap ---------------------------------------------------
    // --- Sun disc brightness (r_sun_disc_brightness) -------------------------------
    // .x = r_sun_disc_brightness disc multiplier; .yzw reserved. sun_extra was full
    // (size / dist-ratio / vflat / planet-radius), so the daytime mirror of the
    // r_moon_disc_brightness disc knob (moon_extra.z) overflows into its own vec4.
    // Scales sunDisc() only -- never the cloud march / NEE lighting.
    float4 sun_extra2;
    // --- end Sun disc brightness ----------------------------------------------
    // --- Wave 9 hosek-sky: cooked coefficients (real dataset) -----------------
    // Cooked per-frame on the CPU from the real Hosek-Wilkie RGB dataset
    // (src/engine/HosekSkyModel.h::Cook, called from Engine.cpp). Read only
    // when sky mode == 3 (hosek). hosek_cfg[i].xyz = the RGB values of shape
    // coefficient i in the reference index order:
    //   [0]=A [1]=B [2]=C [3]=D [4]=E [5]=F [6]=G [7]=I(sqrt-zenith) [8]=H(Mie g).
    // hosek_radiance.xyz = per-channel radiance magnitude L_M, already scaled
    // into engine HDR units (kHosekRadianceScale). .w lanes unused. 10 vec4 =
    // 160 B, appended at the very tail so every existing field offset is
    // untouched. Mirrored in the SPIR-V Frame block below + Engine.cpp PtPush.
    float4 hosek_cfg[9];
    float4 hosek_radiance;
    // --- end Wave 9 hosek-sky cooked coefficients -----------------------------
    // --- Planetary P0 (#254): per-point local up ------------------------------
    // .xyz = planet centre in WORLD space, .w = planet radius in metres.
    // .w <= 0 selects the PLANAR frame -- localUp() is world +Y everywhere
    // and planetAltitude(p) is p.y, i.e. exactly what this shader hard-coded
    // before #254. .w > 0 selects the SPHERICAL frame, where local up is the
    // radial direction from the centre and solar elevation stops being a
    // single global scalar. See localUp() below. 16 B appended at the very
    // tail so every existing field offset is untouched. Mirrored in the
    // SPIR-V Frame block below + Engine.cpp PtPush.
    float4 planet_center_radius;
    // --- end Planetary P0 -----------------------------------------------------
    // --- Planetary P3 (#257): the analytic planet body ------------------------
    // .xyz = ground albedo, .w = the body's radius in metres, or 0 when the
    // body is off. Kept as its own lane rather than folded into
    // planet_center_radius because that field's .w is the ATMOSPHERE shell's
    // radius and is what every altitude query in the shader is measured
    // against -- a scene may legitimately have air and no ground body (every
    // fixture that exists today does), and conflating the two would make
    // "is the ground on?" and "is the frame spherical?" the same question.
    //
    // 16 B appended at the very tail so every existing field offset is
    // untouched. Mirrored in the SPIR-V Frame block below + Engine.cpp PtPush.
    float4 planet_ground;
    // --- end Planetary P3 -----------------------------------------------------
    // --- Planetary P4 (#258): terrain runtime state ---------------------------
    // .x = snowline altitude in metres at this scene's reference site. A
    //      SHADING parameter, not a physical model: the regional snowline is
    //      ~4 900 m in the tropics (Kaser & Osmaston, "Tropical Glaciers",
    //      2002) and drops toward sea level at high latitude, and the engine
    //      scales that anchor by cos(site latitude). Never touches geometry.
    // .y = number of live TLAS instances (informational / debug overlay).
    // .z = 1 when terrain streaming is active this frame, 0 otherwise.
    // .w = reserved.
    //
    // 16 B appended at the very tail so every existing field offset is
    // untouched. Mirrored in the SPIR-V Frame block below + Engine.cpp PtPush.
    float4 planet_terrain;
    // --- end Planetary P4 -----------------------------------------------------
    // --- Planetary P5 (#259): the planetary ocean -----------------------------
    // Six vec4s appended at the tail so every existing field offset is
    // untouched. Mirrored in the SPIR-V Frame block below + Engine.cpp PtPush.
    //
    //   planet_ocean.x = SEA-LEVEL RADIUS in metres, 0 = no planetary ocean.
    //                    The host fills it from the SAME radius that fills
    //                    planet_center_radius.w, for the reason P3 gives
    //                    about the air and the ground: two copies of "where
    //                    is sea level" disagree, and a disagreement between
    //                    the ocean and the terrain is a coastline in the
    //                    wrong place.
    //   planet_ocean.y = number of FFT cascades stacked in the ocean texture
    //                    atlas (1 = the legacy single-tile layout).
    //   planet_ocean.z = Cox & Munk 1954 total mean-square slope for this
    //                    wind speed. The value the BRDF must converge to as
    //                    the ray-cone footprint outgrows every cascade.
    //   planet_ocean.w = reserved.
    //
    //   ocean_tan_e.xyz = tangent-frame EAST basis (unit, render space)
    //   ocean_tan_n.xyz = tangent-frame NORTH basis (unit, render space)
    //   ocean_tan_o.xyz = tangent-frame ORIGIN, render-relative, on the shell
    //   ocean_tan_{e,n,o}.w = cascade 0 / 1 / 2 tile period in metres
    //   ocean_phase.xy  = cascade 1 tile-space phase, .zw = cascade 2's.
    //                     Cascade 0's is 0 by construction -- the anchor
    //                     lattice IS cascade 0's period.
    //   ocean_slope.xyz = each cascade's own total mean-square slope
    //   ocean_slope.w   = alpha^2 of the water microsurface below every
    //                     cascade (0: clean water has no roughness of its
    //                     own, all of it is wave slope).
    float4 planet_ocean;
    float4 ocean_tan_e;
    float4 ocean_tan_n;
    float4 ocean_tan_o;
    float4 ocean_phase;
    float4 ocean_slope;
    // --- end Planetary P5 -----------------------------------------------------
    // --- Planetary P7 (#261): underwater in-scatter ------------------------
    //
    // water_scatter.xyz = the MOLECULAR scattering coefficient of pure
    //                     seawater, 1/m, per RGB channel. Morel 1974's
    //                     beta_w(90 deg) fit through the
    //                     Rayleigh-with-depolarisation sphere integral --
    //                     see src/physics/WaterOptics.h, which derives
    //                     every constant rather than quoting a triple.
    //             .w    = the PARTICULATE scattering coefficient, 1/m,
    //                     grey. Zero is pure seawater, which is the
    //                     default; a Jerlov coastal type is 0.05-0.5.
    //
    // The two are separate because their PHASE FUNCTIONS are separate:
    // molecular scatter is near-symmetric (Rayleigh, depolarisation .w
    // below) and particulate scatter is violently forward (Petzold, g in
    // .x below). Folding them into one lobe is what makes clear water
    // render as tinted glass instead of as water -- pure water's
    // near-isotropic blue scatter is exactly why a clear ocean is blue
    // rather than merely dark.
    float4 water_scatter;
    // water_scatter2.x = Petzold "average particle" asymmetry g (0.924).
    //               .y = in-scatter march sample count.
    //               .z = master gate (0 = no march at all; the frame is
    //                    then pure exponential darkening, which is what
    //                    every build before this one did).
    //               .w = seawater depolarisation ratio (0.09).
    float4 water_scatter2;
    // --- end Planetary P7 --------------------------------------------------
    // --- #280: the cooked sky palette -----------------------------------------
    // procSky's day palette, INTEGRATED on the host once per frame through
    // this scene's own medium rather than authored as literals. Three
    // radiances in W/m^2/sr: the sky at the zenith, and at the horizon in
    // the sun's azimuth and in the anti-solar one.  See
    // src/renderer/Atmosphere.cpp CookSky for what is in them (Rayleigh +
    // ozone + Hillaire 2020 multiple scattering, deliberately NOT the
    // aerosol -- the march owns that lobe).
    //
    //   .w of sky_cook_zenith = 1 when the host cooked this frame, 0 when
    //   it did not (no planet radius, or the sky mode does not need it).
    //   The shader does not branch on it today; it exists so a black sky
    //   can be told apart from an unwritten one.
    //
    // Three vec4 appended at the very tail so every existing field offset
    // is untouched.  Mirrored in the SPIR-V Frame block below +
    // Engine.cpp PtPush.
    float4 sky_cook_zenith;
    float4 sky_cook_horizon_sun;
    float4 sky_cook_horizon_anti;
    // --- end #280 -------------------------------------------------------------
    // --- Land cover (#300): the site's ECEF basis + the raster's shape --------
    //
    // WHY A BASIS AND NOT A LATITUDE. A raster indexed by geodetic
    // (lat, lon) needs the SHADED POINT's coordinates, and the engine's
    // world frame is a local East / Up / South frame pinned at the scene's
    // site (PlanetSite::ecef_to_world, src/renderer/Planet/TerrainChunk.h).
    // Handing the shader the site's latitude -- which is what
    // planet_terrain.x used to carry, folded into a cos() -- gives one
    // number for the whole planet, so a globe seen from orbit had one
    // snowline and could not have had a raster at all. These three rows are
    // the rotation that turns a world-space radial direction back into ECEF,
    // where latitude and longitude exist.
    //
    //   land_basis_e.xyz = the East  row of PlanetSite::ecef_to_world
    //   land_basis_u.xyz = the Up    row  (geocentric, not geodetic -- see
    //                                     TerrainChunk.h on why the centre
    //                                     is geocentric)
    //   land_basis_s.xyz = the South row
    //
    // so ecef = E*w.x + U*w.y + S*w.z for a world-space vector w, which is
    // transpose(ecef_to_world) * w written out.
    //
    //   land_basis_e.w = raster width  in texels, 0 = NO RASTER BOUND. The
    //                    single gate: with 0 here the terrain falls to
    //                    terrainProceduralAlbedo() and land_albedo is never
    //                    read, which is what makes the placeholder binding
    //                    safe.
    //   land_basis_u.w = raster height in texels.
    //   land_basis_s.w = r_planet_land_albedo, the master blend from the
    //                    procedural classification (0) to the raster (1).
    //
    // Three vec4 appended at the very tail so every existing field offset is
    // untouched. Mirrored in the SPIR-V Frame block below + Engine.cpp PtPush.
    float4 land_basis_e;
    float4 land_basis_u;
    float4 land_basis_s;
    // --- end land cover -------------------------------------------------------
#endif  // !PT_TARGET_SPIRV
};

#ifdef PT_TARGET_SPIRV
// Spilled tail of PtPush — see the comments in the (Metal) Push cbuffer
// above for per-field semantics. Layout is byte-equivalent to bytes
// [112..sizeof(PtPush)) of the host PtPush struct in Engine.cpp (832 B
// total today => spilled tail is 720 B; issue #117 grew this by 16 B
// for clouds_p4). Bound by VulkanCommandBuffer::Dispatch from a per-frame
// ring-buffer UBO.
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
    // --- Wave 10 bloom/bokeh --- see the matching field in the Metal
    // Push cbuffer above for per-lane semantics. Layout-equivalent.
    float4 dof_bokeh_params;
    // --- end Wave 10 bloom/bokeh ---
    // --- Wave 8 spectral (#27) --- see the matching field in the Metal
    // Push cbuffer above for per-lane semantics. Layout-equivalent.
    float4 spectral_params;
    // --- end ---
    float4 vol_params;
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 clouds_p4;
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
    // --- SDF Phase 1 (#97) + Phase 3 (#99) ---------------------------------
    // See `sdf_params` / `sdf_params_f` in the Metal Push cbuffer above.
    // .w of uint4 is sdf_fractal_iters; .yz of float4 are
    // r_sdf_fractal_power / r_sdf_de_eps_scale.
    uint4  sdf_params;
    float4 sdf_params_f;
    // --- end SDF Phase 1 + Phase 3 -----------------------------------------
    // See `composite_celestials` in the Metal Push cbuffer above.
    uint composite_celestials;
    uint _pad_celestials0;
    // See `write_shadow_vis` in the Metal Push cbuffer above (issue #115).
    uint write_shadow_vis;
    uint _pad_shadow_vis;
    // See the matching vol_* fields in the Metal Push cbuffer above (#100).
    float vol_density_scale;
    float vol_phase_g_cloud;
    float vol_multiscatter_oct;
    float _pad_vol;
    // See `light_count` / `light_tree_node_count` / `light_tree_enabled`
    // in the Metal Push cbuffer above (#73 + #129). Own 16-byte
    // trailing block (light_count + 3 light-tree uints).
    uint light_count;
    uint light_tree_node_count;
    uint light_tree_enabled;
    uint _pad_light_tail;
    // See `write_specular_albedo_gbuffer` / `write_roughness_gbuffer` /
    // `write_specular_hit_distance_gbuffer` in the Metal Push cbuffer
    // above for semantics (issue #118 MetalFX specular guidance).
    // Layout-equivalent across both targets; trailing pad keeps the
    // 16-byte block alignment for std140 / MSL.
    uint write_specular_albedo_gbuffer;
    uint write_roughness_gbuffer;
    uint write_specular_hit_distance_gbuffer;
    uint _pad_specular_gbuffers0;
    // --- Water Phase 1 (#134) -------------------------------------------------
    // See `water_params0` / `water_params1` in the Metal Push cbuffer above.
    float4 water_params0;
    float4 water_params1;
    // --- end Water Phase 1 ----------------------------------------------------
    // See ReSTIR fields in the Metal Push cbuffer above (#78).
    uint restir_enabled;
    uint restir_k_candidates;
    uint _pad_restir0;
    uint _pad_restir1;
    // See `smoke_params` in the Metal Push cbuffer above (#136).
    float4 smoke_params;
    // See `smoke_params2` in the Metal Push cbuffer above (#180).
    float4 smoke_params2;
    // See `smoke_params3` in the Metal Push cbuffer above (wave-6).
    float4 smoke_params3;
    // See `smoke_sph_params` in the Metal Push cbuffer above (Phase 3, #22).
    float4 smoke_sph_params;
    // See `motion_blur_params` in the Metal Push cbuffer above (#85).
    float4 motion_blur_params;
    // See `water_state` in the Metal Push cbuffer above (Phase 4, #134).
    float4 water_state;
    // See `editor_params` in the Metal Push cbuffer above (agent-19).
    uint4 editor_params;
    // See `clouds_runtime` in the Metal Push cbuffer above (#24, Wave 7).
    uint4 clouds_runtime;
    // See `mesh_motion_prev` / `mesh_motion_curr` in the Metal Push
    // cbuffer above (wave-7 #21).
    float4 mesh_motion_prev;
    float4 mesh_motion_curr;
    // --- Wave 8 ocean (#25) ---------------------------------------------------
    // See `ocean_params0` / `ocean_params1` in the Metal Push cbuffer above.
    float4 ocean_params0;
    float4 ocean_params1;
    // --- end Wave 8 ocean -----------------------------------------------------
    // See `mesh_tex_indices` in the Metal Push cbuffer above (Wave 8 #26).
    uint4 mesh_tex_indices;
    // See `ocean_foam_params` in the Metal Push cbuffer above (Wave 9).
    float4 ocean_foam_params;
    // --- Wave 9 hosek-sky ---
    // See `hosek_params` in the Metal Push cbuffer above.
    float4 hosek_params;
    // --- end Wave 9 hosek-sky ---
    // --- Wave 9 advanced materials --------------------------------------------
    // See `mat_override_count` / `mat_overrides` in the Metal Push cbuffer
    // above. Layout-equivalent across both targets.
    uint4 mat_override_count;
    float4 mat_overrides[8 * 3];
    // --- end Wave 9 advanced materials ----------------------------------------
    // --- Wave 9 tonemap (#27 follow-up) --- see the matching field in the
    // Metal Push cbuffer above. tonemap_params.x = r_tonemap_op enum.
    uint4 tonemap_params;
    // --- end Wave 9 tonemap ---
    // See `sun_extra2` in the Metal Push cbuffer above (r_sun_disc_brightness).
    float4 sun_extra2;
    // See `hosek_cfg` / `hosek_radiance` in the Metal Push cbuffer above
    // (Wave 9 hosek-sky cooked coefficients). Layout-equivalent: 10 vec4
    // appended at the tail. Bytes [tail] of the host PtPush.
    float4 hosek_cfg[9];
    float4 hosek_radiance;
    // See `planet_center_radius` in the Metal Push cbuffer above
    // (Planetary P0, #254). Layout-equivalent: one vec4 appended at the tail.
    float4 planet_center_radius;
    // See `planet_ground` in the Metal Push cbuffer above (Planetary P3,
    // #257). Layout-equivalent: one more vec4 appended at the tail.
    float4 planet_ground;
    // See `planet_terrain` in the Metal Push cbuffer above (Planetary P4,
    // #258). Layout-equivalent: one more vec4 appended at the tail.
    float4 planet_terrain;
    // See `planet_ocean` .. `ocean_slope` in the Metal Push cbuffer above
    // (Planetary P5, #259). Layout-equivalent: six more vec4 at the tail.
    float4 planet_ocean;
    float4 ocean_tan_e;
    float4 ocean_tan_n;
    float4 ocean_tan_o;
    float4 ocean_phase;
    float4 ocean_slope;
    // See `water_scatter` / `water_scatter2` in the Metal Push cbuffer
    // above (Planetary P7, #261).
    float4 water_scatter;
    float4 water_scatter2;
    // See `sky_cook_zenith` .. `sky_cook_horizon_anti` in the Metal Push
    // cbuffer above (#280). Layout-equivalent: three more vec4 at the tail,
    // AFTER P7's pair -- the Frame UBO is byte-equivalent to the host
    // struct, so the two tail appends have to be in the same order on both
    // sides and P7's landed first.
    float4 sky_cook_zenith;
    float4 sky_cook_horizon_sun;
    float4 sky_cook_horizon_anti;
    // See `land_basis_e` .. `land_basis_s` in the Metal Push cbuffer above
    // (land cover, #300). Layout-equivalent: three more vec4 at the tail,
    // AFTER #280's three -- the Frame UBO is byte-equivalent to the host
    // struct, so tail appends have to be in the same order on both sides.
    float4 land_basis_e;
    float4 land_basis_u;
    float4 land_basis_s;
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
// --- SDF Phase 1 (#97) -----------------------------------------------------
// Analytic SDF library (sphere/box/torus/capsule/plane + smooth-CSG
// ops + sphere-trace). Imported as a Slang IR module so body-only
// PathTrace.slang iterations don't re-parse it.
import SdfPrimitives;
// --- end SDF Phase 1 -------------------------------------------------------
// --- SDF Phase 3 (#99) -----------------------------------------------------
// Iconic-fractal DE library (Mandelbulb / Mandelbox / Apollonian).
// Lives in its own module because the trace path is shape-specific
// (no smooth-CSG composition with analytic shapes in Phase 3) and
// normals are recovered by central differences instead of the
// analytic gradient. traceSdfClusters below dispatches to
// traceSdfFractal when a single-leaf cluster carries a fractal shape
// id (>= SDF_FRACTAL_FIRST).
import SdfFractals;
// --- end SDF Phase 3 -------------------------------------------------------
// --- ReSTIR DI Phase A (#78) ----------------------------------------------
// Reservoir struct + WRS streaming primitives (initReservoir,
// streamingUpdate, finalizeReservoir, storeReservoir,
// reservoirTargetPdf). Imported here so the candidate-generation block
// at the primary-hit Lambert NEE site stays a thin readable wrapper
// around the algorithmic helpers. Mirrors the temporal / spatial /
// final kernels that also import this module.
import RestirReservoir;
// --- end ReSTIR DI Phase A ------------------------------------------------

// --- Planetary P0 (#254): per-point local up --------------------------------
//
// "Up" is not a world constant on a planet, and neither is the sun's
// elevation. Everything in this shader's sky / lighting chain used to
// read the single global scalar `sun_and_mode.y` for sin(solar
// elevation) and world +Y for the zenith, which makes time-of-day a
// property of the FRAME rather than of the point being shaded. The
// consequence is not a small error: with one global sun elevation the
// entire world is lit (or unlit) together, so a planet cannot have a lit
// limb and a dark side at the same time -- no terminator is
// representable at all. That is the blocker #51 never named, and it
// gates every later planetary phase.
//
// The fix is mechanical: every such site becomes
// `dot(localUp(P), sun_dir)` for the point P it is actually shading.
//
// PLANAR frame (planet_center_radius.w <= 0, the default): localUp() is
// world +Y and planetAltitude() is p.y, so every rewritten site collapses
// to the exact float it read before. dot(v, float3(0,1,0)) is v.y
// bit-for-bit in IEEE-754 -- 0*v.x and 0*v.z are exactly zero for finite
// inputs and adding them changes nothing -- so this refactor is a
// verified no-op on every existing scene.
//
// SPHERICAL frame: the engine pushes the centre at (0, -R, 0) so the
// y = 0 ground plane stays tangent to the sphere at the world origin --
// the convention atmosphericTransmittance has used for its shell
// integral since #51. Ground geometry, collision and the cloud slab are
// still planar; the curved ground, LUTs and ozone are a later phase.

// Local up at world point p: the outward radial direction from the
// planet centre, or world +Y in the planar frame.
float3 localUp(float3 p) {
    if (planet_center_radius.w <= 0.0) return float3(0.0, 1.0, 0.0);
    return normalize(p - planet_center_radius.xyz);
}

// Altitude of world point p above the surface (metres, can be negative).
// Planar frame: p.y, the engine's ground-plane-at-y=0 convention.
//
// The spherical branch used to be the naive `length(p - c) - R`, which
// suffers the same catastrophic cancellation at planetary radius that
// PathTraceMath's intersectSphere did -- measured at 0.50 m worst case
// over a 100 km ground sweep at R = 6.371e6. Issue #271 moved it onto
// ptAltitudeAboveSphere, which forms the exact |p - c|^2 - R^2 without
// ever differencing p and c and divides by |p - c| + R, a sum that cannot
// cancel: 3.3e-7 m at 1 m altitude, i.e. a few ULPs of the altitude it
// returns. Sign convention is unchanged (negative below the surface),
// and below PathTraceMath's radius gate the historic expression still
// runs verbatim, so a miniature-planet scene renders bit-for-bit as before.
float planetAltitude(float3 p) {
    if (planet_center_radius.w <= 0.0) return p.y;
    return ptAltitudeAboveSphere(p, planet_center_radius.xyz,
                                 planet_center_radius.w);
}

// sin(elevation of `dir` above the local horizon at world point p).
// `dir` must be unit length. This is the per-point replacement for the
// old global `sun_and_mode.y` / `moon_dir_phase.y` scalars.
float elevationAt(float3 p, float3 dir) {
    return dot(localUp(p), dir);
}
// --- end Planetary P0 -------------------------------------------------------

static const uint MAT_LAMBERT    = 0u;
static const uint MAT_METAL      = 1u;
static const uint MAT_DIELECTRIC = 2u;
// MAT_WATER (Phase 1, #134): analytic plane with normal-map waves +
// Schlick Fresnel + Snell refraction + Beer's-law absorption. Driven
// by water_params0 / water_params1 push fields.  Behaves like a
// dielectric for visibility/transmittance but with its own BRDF.
static const uint MAT_WATER      = 3u;

// --- Planetary P7 (#261): the medium stack --------------------------------
//
// WHAT IT REPLACES. `int in_water`, a single bit, set at ray-gen from
// water_state.x and toggled by the MAT_WATER refract branch. MAT_DIELECTRIC
// was deliberately left OUT of that toggle (see the bounce-loop docstring),
// which means a glass body submerged in water kept attenuating with water
// absorption for the whole segment INSIDE the glass. One bit cannot say
// "in water, inside a bubble" -- it can only say water or not-water.
//
// THE REPRESENTATION. Eight two-bit entries in the low 16 bits, the entry
// count in bits 16-19. A uint, so it costs one register and no memory, and
// every operation is a shift and a mask -- this sits in the megakernel's
// hottest loop and in transmittance(), which is why it is not an array.
//
//   bits  0-15 : entries, entry k at bits 2k..2k+1
//   bits 16-19 : depth, 0..8
//   bits 20-31 : reserved, always zero
//
// THE TWO EDGE CASES ARE DEFINED, NOT UNDEFINED. Pushing onto a full stack
// is a no-op that keeps the current top (a ray that has crossed eight
// nested interfaces without leaving one is a pathological scene, and
// saturating is the conservative answer -- it never invents a medium).
// Popping an empty stack yields air. Both are asserted exhaustively in
// tests/pt_water_optics_test.cpp, because a stack that corrupts under
// over/underflow would corrupt the whole path, not just one segment.
//
// BIT-EXACT ABOVE WATER. With the camera in air the stack starts empty, a
// glass entry pushes MEDIUM_DIELECTRIC and its exit pops it, and
// ptMediumIsWater() is false throughout -- the same answer the old
// `in_water == 0` gave, on every branch that reads it.
static const uint MEDIUM_AIR        = 0u;
static const uint MEDIUM_WATER      = 1u;
static const uint MEDIUM_DIELECTRIC = 2u;
static const uint kMediumStackMax   = 8u;

uint ptMediumDepth(uint s) { return (s >> 16) & 0xFu; }

uint ptMediumTop(uint s) {
    uint d = ptMediumDepth(s);
    return (d == 0u) ? MEDIUM_AIR : ((s >> ((d - 1u) * 2u)) & 3u);
}

uint ptMediumPush(uint s, uint id) {
    uint d = ptMediumDepth(s);
    if (d >= kMediumStackMax) return s;         // saturate, never corrupt
    uint entries = (s & 0xFFFFu) & ~(3u << (d * 2u));
    entries |= (id & 3u) << (d * 2u);
    return entries | ((d + 1u) << 16);
}

uint ptMediumPop(uint s) {
    uint d = ptMediumDepth(s);
    if (d == 0u) return s;                      // underflow -> stay in air
    // CLEAR the entry as well as lowering the depth, so the encoding is
    // CANONICAL: one bit pattern per (depth, contents). Leaving the popped
    // entry behind costs nothing to read -- ptMediumTop ignores it -- but it
    // means two stacks that mean the same thing compare unequal, and this
    // value is carried across bounces and into transmittance() where
    // equality is the cheapest way to reason about it.
    uint entries = (s & 0xFFFFu) & ~(3u << ((d - 1u) * 2u));
    return entries | ((d - 1u) << 16);
}

bool ptMediumIsWater(uint s) { return ptMediumTop(s) == MEDIUM_WATER; }

// Enter a medium, IDEMPOTENTLY.
//
// The two notions in play answer different questions. `ndi < 0` is
// GEOMETRIC -- which side of this facet is the ray on -- and the stack is
// TOPOLOGICAL -- which media has it entered and not left. On a displaced
// wave field they disagree occasionally: a ray already inside the water
// that grazes a crest can read "entering", because the crest's shading
// normal faces it. A bare push would take the stack to water-inside-water,
// and the matching exit would pop back to water instead of to air, so the
// ray would keep absorbing as water after it had physically left.
//
// Refusing to re-enter the medium already on top makes the stack agree with
// the geometric sign at every step for a single-medium scene, which is what
// the old one-bit `in_water = 1` did -- and genuine nesting still works,
// because glass inside water has a DIFFERENT id on top and pushes normally.
//
// MEASURED, and small: on golden_water_pool__metal__off, which is the one
// fixture with a wavy surface and submerged bodies, a bare push instead of
// this gives mean-delta 14.313049 where this gives 14.312847 -- two pixels'
// worth, against the same reference. Recorded as the honest size of it
// rather than claimed as a fix: the case is rare, it is a correctness
// property rather than a visible one, and the expensive lesson in this
// phase was in ptMediumCross below, not here.
uint ptMediumEnter(uint s, uint id) {
    return (ptMediumTop(s) == id) ? s : ptMediumPush(s, id);
}

// Cross an interface WITHOUT being told which way. transmittance() has no
// reliable side test -- it computed `ndi` only to orient a normal, and the
// medium bit it kept was flipped unconditionally, `sr_in_water = 1 -
// sr_in_water`, on every MAT_WATER crossing. That is a toggle, and a toggle
// is what a shadow ray needs: it alternates in and out of the body as it
// chains through interfaces, and its FIRST crossing's direction is already
// encoded in the medium the caller handed it.
//
// Re-deriving the direction from `ndi` instead was tried and MEASURED:
// golden_water_pool__metal__off moved by mean-delta 13.41 over 6.95% of
// pixels, concentrated in the far-field wave band -- the shadow rays that
// serve submerged geometry stopped alternating correctly and the pool floor
// near the horizon came back sunlit. The bit's toggle was load-bearing and
// this is its stack form: if the top is already this medium we are leaving
// it, otherwise we are entering.
//
// The camera path does NOT use this. There the sign is real -- ndi_s is
// taken against a shading normal the branch has explicitly oriented -- so
// the refract branches push and pop directionally, exactly as the bit set
// and cleared.
uint ptMediumCross(uint s, uint id) {
    return (ptMediumTop(s) == id) ? ptMediumPop(s) : ptMediumEnter(s, id);
}

// --- #305: the attenuation that governs a REFLECTANCE ----------------------
//
// Which attenuation coefficient a water term uses is the single most
// consequential choice in this medium, and the engine has three different
// answers to it living in three places. This function is the one the
// DIFFUSE terms share.
//
// The full beam attenuation c = a + b is what removes light from an
// IMAGE-FORMING path: a photon scattered out of the line of sight no longer
// contributes to the sea floor's silhouette, which is why water is hazy.
// It is NOT what removes light from a reflectance. A photon scattered 10
// degrees off a downward beam is still going down; one scattered 10 degrees
// off an upwelling ray still reaches the sensor. What removes a photon from
// the upwelling stream is BACKSCATTERING, so the attenuation that governs
// the in-scattered field is
//
//     c_eff = a + b_b
//
// the quasi-single-scattering approximation (H. R. Gordon, "Simple
// calculation of the diffuse reflectance of the ocean", Applied Optics
// 12(12), 2803-2804, 1973), and it is why ocean optics writes K_d ~
// (a + b_b)/mu rather than c/mu. MEASURED, in
// tests/pt_water_optics_test.cpp: against Gordon et al. 1988's published
// r_rs = 0.0949 X + 0.0794 X^2, c_eff puts the engine within 0.5% in the
// blue for pure seawater, where the full c is 14% low -- and at an
// open-ocean particle load the full c is FOUR TIMES low, because it
// charges the reflectance for every forward-scattered photon as if that
// photon were lost.
//
// b_b IS DERIVED FROM THE PHASE FUNCTIONS ALREADY IN USE. The Rayleigh-
// with-depolarisation phase function is even in cos(theta), so its
// backscatter fraction is EXACTLY 1/2 whatever the depolarisation ratio --
// which is the same 1/2 that Smith & Baker 1981 and Morel & Maritorena
// 2001 eq. (11) hard-code for pure seawater. Henyey-Greenstein integrates
// in closed form to B(g) = (1-g)/(2g) [(1+g)/sqrt(1+g^2) - 1], which at
// the Petzold asymmetry 0.924 is 0.016989 against the 0.018 the measured
// Petzold volume scattering function itself integrates to.
//
// Mirrored by pt::water::EffectiveAttenuation in src/physics/WaterOptics.h.
float3 ptWaterBackscatter(float3 b_mol, float b_par, float g) {
    float B_hg = (abs(g) < 1e-6)
                     ? 0.5
                     : (1.0 - g) / (2.0 * g)
                           * ((1.0 + g) / sqrt(1.0 + g * g) - 1.0);
    return max(b_mol, float3(0.0)) * 0.5
         + float3(max(b_par, 0.0) * B_hg);
}

// c_eff for whatever water is in force this frame. Reads the push fields
// directly so no call site can pass a stale triple.
float3 ptWaterEffAtten() {
    return max(water_params0.xyz, float3(0.0))
         + ptWaterBackscatter(water_scatter.xyz, water_scatter.w,
                              clamp(water_scatter2.x, -0.99, 0.99));
}
// --- end #305 --------------------------------------------------------------

// NO ptMediumIor HERE, DELIBERATELY. The obvious companion to this stack is
// a lookup that returns the refractive index of whatever medium is on top,
// so the refract branches can stop assuming the far side of an interface is
// air. It was written, wired in, MEASURED and removed -- see the note in
// the MAT_WATER branch for the measurement and for what would have to
// change before it is worth having. An unused helper sitting here would
// read as an oversight rather than as a decision.
// --- end Planetary P7 medium stack ----------------------------------------

// --- Wave 9 advanced materials --------------------------------------------
// Optional BSDF lobe extension flags (bitmask). Mirrors Engine.h's
// kMatExt* constants. A hit's extension flags come from the push-constant
// mat_overrides array (keyed by prim id), NOT the prim storage buffer, so
// the analytic-prim stride stays 7 float4s. Default 0 = no extension =
// existing material render, bit-for-bit.
static const uint MAT_EXT_ANISOTROPIC = 1u << 0;   // anisotropic GGX (brushed metal)
static const uint MAT_EXT_CLEARCOAT   = 1u << 1;   // Disney clearcoat lobe (car paint)
static const uint MAT_EXT_SUBSURFACE  = 1u << 2;   // random-walk SSS (skin/wax/marble)
// --- end Wave 9 advanced materials ----------------------------------------

static const uint PRIM_SPHERE = 0u;
static const uint PRIM_PLANE  = 1u;

// Light primitive types (#73). Mirrors AnalyticLight::Type on the host.
// Stored as a float in v1.w of each 4-float4 light record because
// StructuredBuffer<float4> can't host mixed-type lanes.
static const uint LIGHT_POINT  = 0u;
static const uint LIGHT_SPOT   = 1u;
static const uint LIGHT_SPHERE = 2u;
static const uint LIGHT_QUAD   = 3u;

static const float3 LIGHT_DIR    = float3(0.4, 1.0, 0.3);
static const float3 MESH_ALBEDO  = float3(1.00, 0.85, 0.45);   // gold metal default

// --- Wave 8 PBR (#26) -- texture atlas sampling ---------------------------
// The atlas is sampled with manual bilinear filtering over integer texel
// fetches, matching the env_map / star_map sampling style already used in
// this shader (there is no SamplerState bound -- the engine's texture
// pipeline is fetch-only, which keeps the cross-backend binding model
// simple). A material's `tile` index selects a kPbrTileSize-square band
// stacked vertically in the atlas; the in-tile UV is wrapped with frac()
// so tiling textures (checkerboards, brick, ...) repeat seamlessly.
//
// sRGB: the atlas stores raw bytes. samplePbrTileLinear() returns them
// verbatim (correct for normal / roughness / metallic data maps);
// samplePbrTileSrgb() applies the sRGB->linear EOTF (correct for albedo
// colour maps). Splitting at the call site is what prevents double-gamma.

// sRGB (IEC 61966-2-1) electro-optical transfer function: encoded ->
// linear. Inverse of the OETF used by Tonemap.slang / linearToSrgb here.
float3 srgbToLinear(float3 c) {
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 0.0), float3(2.4));
    return select(c <= 0.04045, lo, hi);
}

// Atlas origin (x0,y0) + square dimension of mip `level` for a tile band.
// Level 0 is the base image at the band's left; levels 1..kPbrMaxLod are the
// right-column stack. y0 = row_base + (kPbrTileSize - 2*dim) packs each level
// directly below the previous one (128 rows, then 64, ...). MUST mirror
// Engine::BuildPbrTileMips in Engine.cpp.
void pbrMipRegion(uint tile, uint level, out uint x0, out uint y0, out uint dim) {
    uint row_base = tile * kPbrTileSize;
    if (level == 0u) {
        x0 = 0u; y0 = row_base; dim = kPbrTileSize;
    } else {
        dim = kPbrTileSize >> level;                 // 128,64,...,1
        x0  = kPbrTileSize;                           // right column
        y0  = row_base + (kPbrTileSize - 2u * dim);   // vertical stack
    }
}

// Bilinear-fetch one mip `level` of a tile at the (wrapped) tile-local UV.
// Returns the raw RGBA (no colour-space conversion). `tile` MUST be a valid
// (non-kPbrNoTexTile) index; callers gate on that before calling. The in-tile
// wrap is at the LEVEL's resolution so tiling textures repeat seamlessly at
// every mip.
float4 samplePbrTileLevel(uint tile, float2 uv, uint level) {
    uint x0, y0, dim;
    pbrMipRegion(tile, level, x0, y0, dim);
    // Wrap into [0,1) so tiling textures repeat; frac handles negatives.
    float2 fuv = frac(uv);
    // Level-local pixel coordinate (continuous), then split into the four
    // surrounding texels for the bilinear blend.
    float fx = fuv.x * float(dim) - 0.5;
    float fy = fuv.y * float(dim) - 0.5;
    float fx0 = floor(fx);
    float fy0 = floor(fy);
    float tx = fx - fx0;
    float ty = fy - fy0;
    int   D   = int(dim);
    int   ix0 = int(fx0);
    int   iy0 = int(fy0);
    // Wrap the in-level coordinate so the +1 tap at the level edge samples
    // the opposite edge (seamless tiling at every mip).
    int ix1 = ((ix0 + 1) % D + D) % D;
    int iy1 = ((iy0 + 1) % D + D) % D;
    ix0 = (ix0 % D + D) % D;
    iy0 = (iy0 % D + D) % D;
    float4 c00 = pbr_atlas[uint2(x0 + uint(ix0), y0 + uint(iy0))];
    float4 c10 = pbr_atlas[uint2(x0 + uint(ix1), y0 + uint(iy0))];
    float4 c01 = pbr_atlas[uint2(x0 + uint(ix0), y0 + uint(iy1))];
    float4 c11 = pbr_atlas[uint2(x0 + uint(ix1), y0 + uint(iy1))];
    float4 cx0 = lerp(c00, c10, tx);
    float4 cx1 = lerp(c01, c11, tx);
    return lerp(cx0, cx1, ty);
}

// Trilinear-fetch a tile at fractional LOD `lod`: bilinear within the two
// bracketing mip levels, then linear blend between them. lod <= 0 reads
// level 0 verbatim (bit-identical to the pre-mip bilinear fetch), so
// magnified surfaces are unchanged. Returns raw RGBA (no colour conversion).
float4 samplePbrTileLinear(uint tile, float2 uv, float lod) {
    float l  = clamp(lod, 0.0, float(kPbrMaxLod));
    uint  l0 = uint(floor(l));
    uint  l1 = min(l0 + 1u, kPbrMaxLod);
    float f  = l - float(l0);
    float4 a = samplePbrTileLevel(tile, uv, l0);
    if (f <= 0.0) return a;
    float4 b = samplePbrTileLevel(tile, uv, l1);
    return lerp(a, b, f);
}

// Albedo sample: trilinear fetch + sRGB->linear decode. The mip chain is
// box-filtered in the stored (sRGB-encoded) byte space, which is the self-
// consistent choice here because the decode is applied AFTER the trilinear
// blend of raw texels (the same order the level-0 path already used).
float3 samplePbrTileSrgb(uint tile, float2 uv, float lod) {
    return srgbToLinear(samplePbrTileLinear(tile, uv, lod).rgb);
}

// Ray-cone texture LOD (Akenine-Moller et al., "Texture Level of Detail
// Strategies for Real-Time Ray Tracing", Ray Tracing Gems 2019, ch. 20).
// Analytic prims have no automatic ddx/ddy, so we grow a cone of angular
// width `cone_width` (diameter at the hit, = pixel spread * path length) and
// project it onto the surface. The 1/|cos| stretch at grazing angles is what
// kills the horizon aliasing: a receding checkerboard is both far (large
// cone_width) and grazing (small cos), so the footprint -> coarse mip.
// uv_scale is the surface's UV-units-per-metre density (1 for the planar
// map, higher on a sphere); uv_scale <= 0 (meshes / unmapped) -> level 0.
float pbrConeTexLod(float cone_width, float3 rd, float3 n, float uv_scale) {
    if (uv_scale <= 0.0) return 0.0;
    float cos_i  = max(abs(dot(rd, n)), 1e-3);
    float texels = (cone_width / cos_i) * uv_scale * float(kPbrTileSize);
    return clamp(log2(max(texels, 1.0)), 0.0, float(kPbrMaxLod));
}

// Build a tangent-space basis from a geometric normal + the surface UV
// gradient. For analytic prims we don't have per-vertex tangents, so we
// derive an arbitrary-but-stable tangent orthogonal to n (Duff et al.
// 2017 branchless ONB). Mesh hits could supply a UV-derivative tangent
// later; the perturbation strength is small enough that the arbitrary
// frame reads correctly for the test maps. Returns the perturbed
// world-space normal given a tangent-space normal-map sample in [-1,1].
float3 applyNormalMap(float3 n, float3 ts_normal) {
    // Duff branchless orthonormal basis.
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    float3 t = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    float3 bt = float3(b, s + n.y * n.y * a, -n.y);
    float3 perturbed = ts_normal.x * t + ts_normal.y * bt + ts_normal.z * n;
    return normalize(perturbed);
}

// Apply a material's PBR texture maps to a hit in place. Reads the four
// tile indices in h.tex_tiles; for each non-kPbrNoTexTile channel,
// samples the atlas at h.uv and overwrites the corresponding flat value.
// Bit-identical no-op when every tile == kPbrNoTexTile (the default), so
// untextured surfaces keep their flat-color render. Called once per hit,
// right before the BSDF dispatch, so ALL material branches (Lambert /
// metal / dielectric) see the textured values for free.
void applyPbrTextures(inout HitInfo h) {
    // Trilinear LOD for every map (mip-filtered by the ray-cone footprint in
    // h.tex_lod, set at the hit site). All four maps share the same UV +
    // footprint, so they select the same LOD.
    float lod = h.tex_lod;
    // Albedo (sRGB-decoded, multiplies the flat tint so a white flat
    // albedo passes the texture through unchanged).
    if (h.tex_tiles.x != kPbrNoTexTile) {
        h.albedo *= samplePbrTileSrgb(h.tex_tiles.x, h.uv, lod);
    }
    // Roughness (linear; .r scales the flat roughness, so a flat 1.0
    // roughness with a gradient map yields the map directly).
    if (h.tex_tiles.z != kPbrNoTexTile) {
        float r = samplePbrTileLinear(h.tex_tiles.z, h.uv, lod).r;
        h.roughness = saturate(h.roughness * r);
    }
    // Metallic (linear; .r drives a Lambert->metal switch above 0.5).
    // Analytic prims pick their BSDF from the Material enum, but a
    // metallic map lets a Lambert-typed material turn metal where the
    // map is white (parity with glTF metalness; modest scope).
    if (h.tex_tiles.w != kPbrNoTexTile) {
        float m = samplePbrTileLinear(h.tex_tiles.w, h.uv, lod).r;
        if (m > 0.5 && h.mat == MAT_LAMBERT) h.mat = MAT_METAL;
    }
    // Normal map (tangent-space, [0,1]->[-1,1]). Perturb the shading
    // normal LAST so the metallic/roughness reads above used the
    // geometric normal's UV (the UV is independent of the normal anyway).
    if (h.tex_tiles.y != kPbrNoTexTile) {
        float3 ts = samplePbrTileLinear(h.tex_tiles.y, h.uv, lod).rgb * 2.0 - 1.0;
        h.normal = applyNormalMap(h.normal, ts);
    }
}
// --- end Wave 8 PBR -------------------------------------------------------

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
    // Per-prim radiant emission (W/sr per channel). Populated by
    // testAnalyticPrim from primitives[idx*7 + 4].xyz (stride bumped
    // to 7 float4s after #85 motion-blur added v3=prev_pos, #181
    // polish added v4=emission, #206 rotation gizmo added v5=orient,
    // and Wave 8 #26 added v6=PBR texture tiles). Mesh / SDF hit paths reset this to 0 on overwrite
    // -- emission is currently only sourced from analytic-prim hits
    // (#181 polish: emissive rigid bodies via phys_drop_* emission
    // args). The per-bounce loop adds `throughput * h.emission` to
    // radiance BEFORE BRDF eval, matching the additive-emitter pattern
    // used for emissive sphere/quad area lights.
    float3 emission;
    // Editor backend (agent-19): user-facing id of the analytic prim
    // that produced this hit. Sourced from primitives[idx*7 + 2].w
    // (the same v2.w slot the host now bakes the AnalyticPrim map key
    // into via asuint reinterpretation). 0 on mesh / SDF / miss paths
    // -- those overwrite the field explicitly. The per-bounce loop
    // tints emission with magenta at silhouette edges when this id
    // matches editor_params.x.
    uint prim_id;
    // --- Wave 8 PBR (#26) -----------------------------------------------
    // Surface UV at the hit (procedural for analytic prims: sphere
    // lat/long, plane world-XZ; barycentric-interpolated for meshes).
    // Used to sample the PBR texture atlas. Defaults to (0,0) on the
    // miss / SDF paths (they don't sample textures).
    float2 uv;
    // Per-material PBR texture tile indices into the strip atlas. uint
    // kPbrNoTexTile (0xFFFFFFFF) = no texture, use the flat values above.
    // x=albedo (sRGB-decoded), y=normal (tangent-space), z=roughness,
    // w=metallic. All default to kPbrNoTexTile so non-textured hits skip
    // the sample entirely and render bit-identically to the flat path.
    uint4 tex_tiles;
    // Ray-cone mip LOD support. uv_scale is the surface's UV-units-per-metre
    // density, set by the analytic-prim hit paths (1 for planes, lat/long
    // density for spheres); 0 on mesh / SDF / miss paths -> base level. tex_lod
    // is the fractional mip level, computed at the hit site from the cone
    // footprint (pbrConeTexLod) and consumed by applyPbrTextures. Both default
    // to 0 so untextured hits are unaffected.
    float uv_scale;
    float tex_lod;
    // --- end Wave 8 PBR -------------------------------------------------
    // --- Wave 9 advanced materials --------------------------------------
    // BSDF lobe extension flags + params for this hit, resolved from the
    // push-constant mat_overrides array by prim id (see resolveMatExt).
    // mat_ext == 0 (the default for every prim without an override slot)
    // means no extension lobe -- the material renders exactly as before.
    uint   mat_ext;          // MAT_EXT_* bitmask
    float  aniso_amount;     // [-1,1] anisotropy (0 = isotropic)
    float  aniso_rotation;   // tangent-frame rotation about the normal (rad)
    float  clearcoat_weight; // [0,1] clearcoat lobe strength
    float  clearcoat_rough;  // [0,1] clearcoat microfacet roughness
    float  subsurface_radius;// scattering mean free path (m)
    float3 subsurface_color; // per-channel single-scatter albedo
    // --- end Wave 9 advanced materials ----------------------------------
};

// Forward declaration: the subsurface random walk (below) re-traces the
// scene to find the medium boundary, but traceScene is defined further
// down. Declare it here so the SSS helper links.
HitInfo traceScene(float3 ro, float3 rd, float shutter_t01, PtRayCone cone);

// --- Wave 9 advanced materials: per-hit extension lookup ------------------
// Resolve the BSDF lobe extension for a hit by scanning the push-constant
// mat_overrides array for a slot whose prim id matches h.prim_id. Called
// once per hit (after applyPbrTextures, before the BSDF dispatch). When
// mat_override_count.x == 0 -- the case for every scene with no advanced
// material -- the loop body never runs, so this is a single compare and
// the existing materials stay bit-for-bit. The match is by USER prim id
// (stable across the host's BVH reorder), so it works regardless of where
// the prim landed in the storage buffer.
void resolveMatExt(inout HitInfo h) {
    uint n = min(mat_override_count.x, 8u);
    if (n == 0u || h.prim_id == 0u) return;
    [loop]
    for (uint s = 0u; s < n; ++s) {
        float4 a = mat_overrides[s * 3u + 0u];
        if (asuint(a.x) != h.prim_id) continue;
        float4 b = mat_overrides[s * 3u + 1u];
        float4 c = mat_overrides[s * 3u + 2u];
        h.mat_ext           = asuint(a.y);
        h.aniso_amount      = a.z;
        h.aniso_rotation    = a.w;
        h.clearcoat_weight  = b.x;
        h.clearcoat_rough   = b.y;
        h.subsurface_radius = b.z;
        h.subsurface_color  = c.xyz;
        return;
    }
}

// --- Wave 9 anisotropic GGX (brushed metal) -------------------------------
// Heitz 2014 "Understanding the Masking-Shadowing Function in Microfacet-
// Based BRDFs" anisotropic GGX. The microfacet normal distribution and the
// Smith height-correlated masking-shadowing term both take separate
// roughnesses (alpha_x, alpha_y) along the tangent / bitangent. We sample
// the visible-normal distribution (VNDF, Heitz 2018 JCGT) so the importance
// sample matches the masking term -- this keeps the estimator low-variance
// at grazing angles and energy-conserving.
//
// Build an orthonormal tangent frame (t, bt, n) from the shading normal and
// a rotation about it. The reference tangent is the Duff et al. 2017
// branchless basis (same one applyNormalMap uses); aniso_rotation then spins
// (t, bt) about n so the brushed streak can point any direction. For UV-
// derived tangents on textured meshes a future change can pass a real
// tangent here; the arbitrary-but-stable frame is correct for analytic
// prims (the highlight is rotation-symmetric about n until anisotropy bends
// it, and aniso_rotation gives explicit control).
void anisoTangentFrame(float3 n, float rot, out float3 t, out float3 bt) {
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float bb = n.x * n.y * a;
    float3 t0 = float3(1.0 + s * n.x * n.x * a, s * bb, -s * n.x);
    float3 bt0 = float3(bb, s + n.y * n.y * a, -n.y);
    float cr = cos(rot);
    float sr = sin(rot);
    t  = normalize(cr * t0 + sr * bt0);
    bt = normalize(cross(n, t));
}

// Map an isotropic roughness + anisotropy in [-1,1] to (alpha_x, alpha_y).
// aspect = sqrt(1 - 0.9 * |aniso|) follows the Disney convention (the 0.9
// cap keeps alpha non-zero so the highlight never collapses to a delta).
// Positive aniso stretches along the tangent (alpha_x large), negative along
// the bitangent. Roughness is squared to alpha (perceptual-linear remap,
// matching the isotropic GGX convention used elsewhere).
void anisoAlpha(float roughness, float aniso, out float ax, out float ay) {
    float r = clamp(roughness, 0.02, 1.0);
    float alpha = r * r;
    float a = clamp(aniso, -1.0, 1.0);
    float aspect = sqrt(1.0 - 0.9 * abs(a));
    float ax_ = alpha / aspect;
    float ay_ = alpha * aspect;
    if (a < 0.0) { float tmp = ax_; ax_ = ay_; ay_ = tmp; }
    ax = clamp(ax_, 0.001, 1.0);
    ay = clamp(ay_, 0.001, 1.0);
}

// Sample the GGX VNDF for a view direction expressed in the local tangent
// frame (Heitz 2018, "Sampling the GGX Distribution of Visible Normals").
// `ve` is the (positive-hemisphere) view direction in local space where
// z = normal. Returns the sampled micro-normal in local space.
float3 sampleGgxVndfLocal(float3 ve, float ax, float ay, float u1, float u2) {
    // Section 3.2: stretch the view to the hemisphere config.
    float3 vh = normalize(float3(ax * ve.x, ay * ve.y, ve.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq)
                            : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(u1);
    float phi = 6.28318530717958 * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float sc = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - sc) * sqrt(max(0.0, 1.0 - p1 * p1)) + sc * p2;
    float3 nh = p1 * t1 + p2 * t2 +
                sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    // Unstretch back to the ellipsoid config.
    return normalize(float3(ax * nh.x, ay * nh.y, max(0.0, nh.z)));
}

// Smith height-correlated masking-shadowing G2 for anisotropic GGX, plus
// the Lambda terms (Heitz 2014 eq. 72/99). l, v are local-space (z=normal).
float ggxLambdaAniso(float3 w, float ax, float ay) {
    // w is normalized; guard the degenerate w.z ~ 1 case.
    float az2 = w.z * w.z;
    if (az2 >= 0.9999) return 0.0;
    float t2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / max(az2, 1e-8);
    return 0.5 * (-1.0 + sqrt(1.0 + t2));
}
float ggxG2Aniso(float3 l, float3 v, float ax, float ay) {
    return 1.0 / (1.0 + ggxLambdaAniso(l, ax, ay) + ggxLambdaAniso(v, ax, ay));
}

// Importance-sample the anisotropic GGX specular lobe and return the
// outgoing (reflected) direction in WORLD space plus the BRDF weight to
// multiply into throughput. The VNDF sample makes the weight collapse to
// the masking ratio G2(l,v)/G1(v) -- the standard low-variance microfacet
// estimator -- with the Fresnel folded in by the caller (metal uses the
// albedo as F0). `wo` is the direction FROM the surface toward the viewer
// (i.e. -rd). On a back-facing or degenerate sample, returns weight 0 so
// the caller can terminate the path.
struct AnisoSample {
    float3 dir;     // world-space reflected direction (toward next bounce)
    float3 weight;  // multiply into throughput (includes F0 + masking)
    bool   valid;
};
AnisoSample sampleAnisoGgx(float3 n, float3 wo, float3 f0,
                           float roughness, float aniso, float rot,
                           inout uint seed) {
    AnisoSample r;
    r.dir = reflect(-wo, n);
    r.weight = float3(0.0);
    r.valid = false;
    float ax, ay;
    anisoAlpha(roughness, aniso, ax, ay);
    float3 t, bt;
    anisoTangentFrame(n, rot, t, bt);
    // World -> local (z = n). wo into local space.
    float3 vl = float3(dot(wo, t), dot(wo, bt), dot(wo, n));
    if (vl.z <= 0.0) return r;
    float u1 = randf(seed);
    float u2 = randf(seed);
    float3 hl = sampleGgxVndfLocal(vl, ax, ay, u1, u2);
    // Reflect the local view about the sampled micro-normal.
    float3 ll = reflect(-vl, hl);
    if (ll.z <= 0.0) return r;
    // VNDF estimator weight: F * G2(l,v) / G1(v). With G1(v) =
    // 1/(1+Lambda(v)), the ratio is (1+Lambda(v)) * G2.
    float g2 = ggxG2Aniso(ll, vl, ax, ay);
    float lam_v = ggxLambdaAniso(vl, ax, ay);
    float g1v_inv = 1.0 + lam_v;     // 1 / G1(v)
    // Fresnel at the micro-facet (Schlick, per-channel F0). cos = dot(v, h).
    float voh = clamp(dot(vl, hl), 0.0, 1.0);
    float fc = pow(1.0 - voh, 5.0);
    float3 F = f0 + (float3(1.0) - f0) * fc;
    r.weight = F * g2 * g1v_inv;
    // Local -> world.
    r.dir = normalize(ll.x * t + ll.y * bt + ll.z * n);
    r.valid = true;
    return r;
}
// --- end Wave 9 anisotropic GGX -------------------------------------------

// --- Wave 9 clearcoat (car paint / lacquer) -------------------------------
// A second specular lobe layered on top of the base BSDF: a thin dielectric
// coat (IOR 1.5, F0 = ((1.5-1)/(1.5+1))^2 = 0.04) with its own roughness.
// Standard layered model evaluated stochastically -- at a hit we reflect off
// the coat with probability p = clearcoat_weight * Fresnel(coat), otherwise
// the ray transmits through the coat and the base material shades it. This
// keeps the estimator unbiased and energy-conserving (the coat lobe's
// throughput weight is F/p == 1/clearcoat_weight when chosen, the base
// material's bounce just continues) and composes with ANY base BSDF
// (lambert / metal / dielectric). The coat is a clear lacquer so it
// reflects the environment independently of the base colour -- the gloss
// flake-over-pigment look of automotive paint.
//
// kClearcoatF0 is the air->lacquer Schlick base reflectance. The coat
// reflection uses an isotropic GGX VNDF sample (reuse the anisotropic
// machinery with ax == ay) so the gloss highlight is sharp at low coat
// roughness and broadens smoothly as roughness rises.
static const float kClearcoatF0  = 0.04;
static const float kClearcoatIOR = 1.5;

// Schlick Fresnel reflectance of the clearcoat at the given view cosine.
float clearcoatFresnel(float cos_v) {
    float fc = pow(1.0 - clamp(cos_v, 0.0, 1.0), 5.0);
    return kClearcoatF0 + (1.0 - kClearcoatF0) * fc;
}

// Sample the clearcoat reflection lobe. Returns the world-space reflected
// direction + the BRDF weight (G2/G1 masking ratio; Fresnel + the 1/p
// selection weight are folded in by the caller). wo points toward the
// viewer (-rd). valid == false on a below-horizon sample.
struct CoatSample {
    float3 dir;
    float3 weight;   // masking ratio (scalar, broadcast)
    bool   valid;
};
CoatSample sampleClearcoat(float3 n, float3 wo, float roughness,
                           inout uint seed) {
    CoatSample r;
    r.dir = reflect(-wo, n);
    r.weight = float3(0.0);
    r.valid = false;
    float a;
    {
        float ax, ay;
        anisoAlpha(roughness, 0.0, ax, ay);   // isotropic: ax == ay
        a = ax;
    }
    // Local frame (z = n) via the Duff ONB (rotation 0).
    float3 t, bt;
    anisoTangentFrame(n, 0.0, t, bt);
    float3 vl = float3(dot(wo, t), dot(wo, bt), dot(wo, n));
    if (vl.z <= 0.0) return r;
    float u1 = randf(seed);
    float u2 = randf(seed);
    float3 hl = sampleGgxVndfLocal(vl, a, a, u1, u2);
    float3 ll = reflect(-vl, hl);
    if (ll.z <= 0.0) return r;
    float g2 = ggxG2Aniso(ll, vl, a, a);
    float g1v_inv = 1.0 + ggxLambdaAniso(vl, a, a);
    r.weight = float3(g2 * g1v_inv);
    r.dir = normalize(ll.x * t + ll.y * bt + ll.z * n);
    r.valid = true;
    return r;
}
// --- end Wave 9 clearcoat -------------------------------------------------

// --- Wave 9 subsurface scattering (skin / wax / marble) -------------------
// Bounded random-walk SSS: the physically-grounded choice for a path
// tracer. A ray entering a translucent material refracts (matched-IOR here,
// so it transmits straight in), then takes free-flight steps through the
// medium -- each step length is exponentially distributed by the scattering
// coefficient sigma_s = 1 / mean-free-path -- bouncing off a Henyey-
// Greenstein phase function at each scatter event and losing energy to the
// per-channel single-scatter albedo (subsurface_color: reds survive deeper
// than blues, the hallmark of skin / wax). When a free flight would carry
// the ray past the medium boundary the ray EXITS there and resumes the
// normal bounce loop in air. The walk is capped at kSssMaxSteps so deep
// absorption terminates cleanly instead of looping forever.
//
// Mean free path is in metres (skin ~1-3 mm, wax ~5-20 mm, marble ~3-10
// mm), matching the engine's 1-unit-per-metre convention. The boundary is
// found by re-tracing the scene from inside the medium -- for a convex prim
// (sphere) intersectSphere returns the far-wall exit distance, so a single
// isolated SSS body diffuses correctly.
static const uint kSssMaxSteps = 24u;

// Sample a scattered direction from the Henyey-Greenstein phase function
// about the incoming direction `wi` (the direction of travel). g in
// (-1, 1): 0 = isotropic, >0 forward-scattering (skin / wax are mildly
// forward, g ~ 0.2..0.8), <0 back-scattering. Standard inverse-CDF form.
float3 sampleHenyeyGreenstein(float3 wi, float g, inout uint seed) {
    float u1 = randf(seed);
    float u2 = randf(seed);
    float cos_t;
    if (abs(g) < 1e-3) {
        cos_t = 1.0 - 2.0 * u1;        // isotropic
    } else {
        float sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cos_t = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    cos_t = clamp(cos_t, -1.0, 1.0);
    float sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    float phi = 6.28318530717958 * u2;
    // Build an orthonormal basis around wi and rotate.
    float3 w = normalize(wi);
    float3 ax = abs(w.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(w, ax));
    float3 b = cross(w, t);
    return normalize(sin_t * cos(phi) * t + sin_t * sin(phi) * b + cos_t * w);
}

// Result of the random walk: the exit ray origin / direction (in air,
// ready for the normal bounce loop) plus the throughput tint accumulated
// from the per-scatter single-scatter albedo. `escaped` is true when the
// ray exited the medium (continue the bounce loop); false when the walk hit
// the step cap deep inside (the path is absorbed -- terminate).
struct SssWalk {
    float3 ro;
    float3 rd;
    float3 tint;     // multiply into throughput
    bool   escaped;
};
SssWalk subsurfaceRandomWalk(float3 entry_pt, float3 in_dir, float3 normal,
                             float mfp, float3 albedo, float shutter_t01,
                             PtRayCone cone, inout uint seed) {
    SssWalk w;
    w.tint = float3(1.0);
    w.escaped = false;
    // Step just inside the surface along the incoming direction (matched-
    // IOR transmission -- the translucent diffusion look doesn't need the
    // refractive bend, and skipping it avoids TIR trapping the walk).
    float3 nf = (dot(normal, in_dir) < 0.0) ? normal : -normal;
    // Step just INSIDE the surface: the same scale-relative offset every
    // other origin gets (#256), taken along -nf instead of +nf.
    float3 ro = ptRayOrigin(entry_pt, -nf, cone, in_dir);
    float3 rd = in_dir;
    // Scattering coefficient (1/m). Guard a tiny mfp so sigma stays finite.
    float sigma_s = 1.0 / max(mfp, 1e-4);
    // Mildly forward-scattering phase (skin / wax). 0.3 is a reasonable
    // tissue-like default; the look is dominated by mfp + albedo anyway.
    const float g_sss = 0.3;
    w.ro = ro;
    w.rd = rd;
    [loop]
    for (uint i = 0u; i < kSssMaxSteps; ++i) {
        // Free-flight distance to the next scatter event.
        float u = randf(seed);
        float d = -log(max(1e-6, 1.0 - u)) / sigma_s;
        // Distance to the medium boundary along rd (re-trace the scene;
        // from inside a convex prim this is the far-wall exit).
        HitInfo bh = traceScene(ro, rd, shutter_t01, cone);
        float t_exit = bh.hit ? bh.t : 1e30;
        if (d < t_exit) {
            // Scatter event inside the medium. Lose energy to the single-
            // scatter albedo, then pick a new HG direction and continue.
            ro = ro + rd * d;
            w.tint *= albedo;
            rd = sampleHenyeyGreenstein(rd, g_sss, seed);
            // Russian-roulette the walk once the tint dims so deep, dark
            // media don't burn the full step budget every path.
            float pr = max(w.tint.r, max(w.tint.g, w.tint.b));
            if (i >= 3u) {
                pr = clamp(pr, 0.02, 1.0);
                if (randf(seed) > pr) { w.escaped = false; return w; }
                w.tint /= pr;
            }
        } else {
            // Reached the boundary -- exit into air just past the surface
            // and hand control back to the bounce loop. The exit direction
            // is the (HG-randomized) walk direction, which is what gives
            // the soft, depth-tinted translucency.
            float3 exit_pt = ro + rd * t_exit;
            float3 exit_n = bh.hit ? bh.normal : nf;
            // Nudge outside along the OUTWARD normal at the exit (the exit
            // normal points along the wall; pick the side the ray leaves).
            float3 eo = (dot(exit_n, rd) > 0.0) ? exit_n : -exit_n;
            w.ro = ptRayOrigin(exit_pt, eo, ptConeAdvance(cone, t_exit), rd);
            w.rd = rd;
            w.escaped = true;
            return w;
        }
    }
    // Hit the step cap deep inside the medium -- treat as absorbed.
    w.escaped = false;
    return w;
}
// --- end Wave 9 subsurface scattering -------------------------------------

// Rotate a 3-vector by a unit quaternion (xyzw). Standard
// q * (0, v) * q^-1 in optimised cross-product form: equals
//   v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
// Used by the plane intersect path so a rotate-gizmo dispatch to
// the host actually tilts the rendered plane. Identity quat (0,0,0,1)
// is a no-op: cross(zero, anything) == 0, so rotated == v exactly.
float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

// Test a single analytic primitive at `primitives[idx*7 + 0..6]` against
// (ro, rd) and update `h` if it's a closer hit. Centralizes the per-prim
// dispatch so the linear-loop and BVH-leaf paths share one body.
//
// Stride is 7 float4s (28 floats) after #85 (motion blur, v3=prev_pos),
// #181 polish (emission, v4), #206 rotation gizmo (orient, v5), and
// Wave 8 #26 (PBR texture tiles, v6):
//   v0 = pos / radius   v1 = albedo / roughness   v2 = type/mat/ior/id
//   v3 = prev_pos / pad (motion blur)
//   v4 = emission / pad (self-emissive rigid bodies)
//   v5 = orient quaternion (rotation gizmo)
//   v6 = albedo/normal/roughness/metallic texture tile indices (PBR #26;
//        uint via asuint, kPbrNoTexTile == 0xFFFFFFFF means no texture)
// Must stay in sync with Engine.cpp's kFloatsPerPrim = 28.
//
// Motion blur (#85): `shutter_t01` is the per-ray shutter sample in
// [0, 1] -- 0 = start of shutter (== prev_pos), 1 = end (== curr_pos =
// v0.xyz). When the host's motion-blur gate is off it forces
// shutter_t01 = 1.0 so the sphere center collapses to v0.xyz
// (bit-identical to pre-#85 behaviour). Planes don't move so they
// ignore the shutter-time param.
//
// Emission (#181 polish): v4.rgb feeds h.emission unconditionally;
// non-emissive prims have v4.rgb == 0 so the cost is just one load.
// The per-bounce loop adds throughput * h.emission to radiance BEFORE
// BRDF eval.
//
// Orientation (#206): spheres are rotation-symmetric so v5 is ignored
// for PRIM_SPHERE. For PRIM_PLANE we rotate v0.xyz (the stored normal)
// by the quaternion to get the effective world-space normal, and the
// plane equation becomes (rotated_normal . p + d == 0). The default
// identity quat (0,0,0,1) is a no-op in quatRotate so existing planes
// render bit-identically.
void testAnalyticPrim(uint idx, float3 ro, float3 rd, float shutter_t01,
                      float t_min, inout HitInfo h) {
    float4 v0 = primitives[idx * 7u + 0u];
    float4 v1 = primitives[idx * 7u + 1u];
    float4 v2 = primitives[idx * 7u + 2u];
    float4 v3 = primitives[idx * 7u + 3u];
    float4 v4 = primitives[idx * 7u + 4u];
    float4 v5 = primitives[idx * 7u + 5u];
    // Wave 8 PBR (#26): v6 holds the 4 texture tile indices (uint via
    // asuint). kPbrNoTexTile == 0xFFFFFFFF on every channel for the
    // default (untextured) prim, so reading them is one extra load and
    // the shading path's sample gate skips when all are kPbrNoTexTile.
    float4 v6 = primitives[idx * 7u + 6u];
    uint4 tex_tiles = uint4(asuint(v6.x), asuint(v6.y),
                            asuint(v6.z), asuint(v6.w));
    uint type = uint(v2.x);
    // v2.w carries the user-facing prim id bit-reinterpreted as float
    // by the host (Engine::EnsurePrimitivesUploaded). The editor
    // outline path reads this through HitInfo::prim_id to compare
    // against editor_params.x. Non-editor frames pay just one extra
    // asuint() per hit -- negligible vs the BRDF / NEE downstream.
    uint user_id = asuint(v2.w);
    float t;
    if (type == PRIM_SPHERE) {
        // Lerp from prev_pos (v3.xyz) toward curr_pos (v0.xyz) at the
        // shutter-time sample. The engine snapshots curr -> prev once
        // per StepPhysics tick so the per-frame displacement is encoded
        // in (v0.xyz - v3.xyz).
        float3 sphere_center = lerp(v3.xyz, v0.xyz, shutter_t01);
        if (intersectSphere(ro, rd, sphere_center, v0.w, t_min, t) && t < h.t) {
            h.hit = true; h.t = t;
            float3 sn = normalize(ro + rd * t - sphere_center);
            h.normal = sn;
            h.mat = uint(v2.y);
            h.albedo = v1.rgb;
            h.roughness = v1.a;
            h.ior = v2.z;
            h.emission = v4.rgb;
            h.prim_id = user_id;
            // Wave 8 PBR (#26): procedural lat/long UV for spheres.
            // u = longitude in [0,1) from atan2 (seam at -x), v = latitude
            // in [0,1] from asin (0 at south pole, 1 at north). Matches the
            // common equirectangular sphere mapping so an equirect texture
            // wraps without distortion at the equator.
            const float kPI = 3.14159265358979;
            h.uv = float2(0.5 + atan2(sn.z, sn.x) / (2.0 * kPI),
                          0.5 - asin(clamp(sn.y, -1.0, 1.0)) / kPI);
            h.tex_tiles = tex_tiles;
            // Ray-cone LOD density (UV units per metre along the surface).
            // Latitude is uniform: dv/ds = 1/(pi*R). Longitude is denser near
            // the poles: du/ds = 1/(2*pi*R*cos(lat)); clamp cos(lat) so the
            // pole singularity can't force an absurd LOD. Take the max axis so
            // the coarser-needed direction wins (conservative vs aliasing).
            float sphere_R = max(v0.w, 1e-4);
            float cos_lat  = max(length(sn.xz), 0.05);
            h.uv_scale = max(1.0 / (kPI * sphere_R),
                             1.0 / (2.0 * kPI * sphere_R * cos_lat));
        }
    } else if (type == PRIM_PLANE) {
        // Rotate the stored normal by the orientation quaternion. With
        // the default identity orient (0,0,0,1) the cross-product form
        // collapses to v0.xyz so untouched planes are bit-identical.
        float3 plane_n = quatRotate(v5, v0.xyz);
        if (intersectPlane(ro, rd, plane_n, v0.w, t_min, t) && t < h.t) {
            h.hit = true; h.t = t;
            h.normal = plane_n;
            h.mat = uint(v2.y);
            h.albedo = v1.rgb;
            h.roughness = v1.a;
            h.ior = v2.z;
            h.emission = v4.rgb;
            h.prim_id = user_id;
            // Wave 8 PBR (#26): procedural planar UV from the world-space
            // hit point. Project onto the two axes most orthogonal to the
            // plane normal so an arbitrarily-oriented plane still gets a
            // sensible 2D parameterization; 1 UV unit = 1 metre (the
            // texture tiles every metre via the shader's frac() wrap).
            // Default-up floor (normal = +y) reduces to (x, z).
            float3 hp = ro + rd * t;
            float3 an = abs(plane_n);
            float2 puv;
            if (an.y >= an.x && an.y >= an.z) {
                puv = float2(hp.x, hp.z);          // horizontal plane: XZ
            } else if (an.x >= an.z) {
                puv = float2(hp.z, hp.y);          // x-facing: ZY
            } else {
                puv = float2(hp.x, hp.y);          // z-facing: XY
            }
            h.uv = puv;
            h.tex_tiles = tex_tiles;
            // Planar UV is isometric (1 UV unit = 1 metre) regardless of the
            // plane's orientation, so the texel density is exactly one UV unit
            // per metre. This is the surface that aliases at the horizon.
            h.uv_scale = 1.0;
        }
    }
}

// --- SDF Phase 1 (#97) -----------------------------------------------------
// Per-cluster stride in float4s: 4 (header) + kMaxNodes * 3. Header
// layout:
//   hdr0.xyz = aabb_min ;  asuint(hdr0.w) = node_count
//   hdr1.xyz = aabb_max ;  asuint(hdr1.w) = material (0/1/2)
//   hdr2.xyz = albedo   ;          hdr2.w = roughness
//   hdr3.x   = ior      ;  hdr3.yzw = pad
// Each SdfClusterNode occupies 3 float4s:
//   n0.x = asfloat(op) ; n0.y = asfloat(shape) ;
//   n0.z = asfloat(child_a) ; n0.w = asfloat(child_b)
//   n1   = params (float4)
//   n2.xyz = center ; n2.w pad
static const uint kSdfHeaderFloat4s = 4u;
static const uint kSdfNodeFloat4s   = 3u;
static const uint kSdfClusterStride = kSdfHeaderFloat4s + SDF_MAX_NODES * kSdfNodeFloat4s; // 28

SdfCluster loadSdfCluster(uint cluster_idx) {
    uint base = cluster_idx * kSdfClusterStride;
    SdfCluster c;
    float4 h0 = sdf_clusters[base + 0u];
    float4 h1 = sdf_clusters[base + 1u];
    float4 h2 = sdf_clusters[base + 2u];
    float4 h3 = sdf_clusters[base + 3u];
    c.aabb_min   = h0.xyz;
    c.node_count = asuint(h0.w);
    c.aabb_max   = h1.xyz;
    c.material   = asuint(h1.w);
    c.albedo     = h2.xyz;
    c.roughness  = h2.w;
    c.ior        = h3.x;
    [unroll(SDF_MAX_NODES)]
    for (uint i = 0u; i < SDF_MAX_NODES; ++i) {
        uint nb = base + kSdfHeaderFloat4s + i * kSdfNodeFloat4s;
        float4 n0 = sdf_clusters[nb + 0u];
        float4 n1 = sdf_clusters[nb + 1u];
        float4 n2 = sdf_clusters[nb + 2u];
        SdfClusterNode n;
        n.op      = asuint(n0.x);
        n.shape   = asuint(n0.y);
        n.child_a = asuint(n0.z);
        n.child_b = asuint(n0.w);
        n.params  = n1;
        n.center  = n2.xyz;
        n._pad    = 0u;
        c.nodes[i] = n;
    }
    return c;
}

// Test all SDF clusters against the ray and update the hit record with
// any closer hit found. Linear scan -- Phase 1 expects <= 8 clusters,
// per the issue's perf budget. A future phase will fold this into the
// analytic-prim BVH as a third leaf-shape tag.
//
// Phase 3 (#99): dispatches fractal-leaf clusters to traceSdfFractal
// (SdfFractals.slang). A cluster is treated as a fractal when its
// root LEAF node carries a shape id >= SDF_FRACTAL_FIRST. The
// dispatcher only handles single-leaf fractal clusters; mixed
// fractal/analytic CSG compositions are scope-out for Phase 3 and
// the host's `combine` helper rejects them at parse time.
void traceSdfClusters(float3 ro, float3 rd, float t_min, inout HitInfo h) {
    uint count = sdf_params.x;
    if (count == 0u) return;
    uint  iter_budget = sdf_params.y;
    float epsilon     = sdf_params_f.x;
    // --- SDF Phase 3 (#99) cvar unpack -------------------------------------
    // Fractal-side knobs live in the otherwise-reserved push lanes;
    // see the Push cbuffer above for the wire convention.
    uint  fractal_iters    = max(sdf_params.w, 1u);
    float fractal_power    = sdf_params_f.y;
    float fractal_de_eps_s = sdf_params_f.z;
    // --- end SDF Phase 3 ---------------------------------------------------
    // Phase 2 (#98): r_sdf_normal_mode now lives at sdf_params_f.w because
    // Phase 3 (#99) claimed sdf_params.w first. Encoded as float (0.0 = forward
    // AD, 1.0 = central differences); the > 0.5 test resolves the toggle.
    uint  normal_mode = (sdf_params_f.w > 0.5f) ? 1u : 0u;
    for (uint i = 0u; i < count; ++i) {
        SdfCluster c = loadSdfCluster(i);
        float t;
        float3 n;
        // --- SDF Phase 3 (#99) fractal dispatch ----------------------------
        // Check the root node (last in post-order) to decide which
        // trace path to take. A single-leaf cluster (node_count == 1)
        // whose shape id is in the fractal range goes through the
        // Phase 3 fractal trace; everything else takes the analytic
        // path. node_count==0 is a malformed cluster -- the analytic
        // path's sphereCluster trace returns no-hit so the cluster is
        // skipped harmlessly.
        bool is_fractal = false;
        if (c.node_count > 0u) {
            SdfClusterNode root = c.nodes[c.node_count - 1u];
            is_fractal = (root.op == SDF_OP_LEAF) &&
                         (root.shape >= SDF_FRACTAL_FIRST);
        }
        if (is_fractal) {
            SdfClusterNode root = c.nodes[c.node_count - 1u];
            // root.center is in cluster-local space (the host packs
            // the per-node translation into n.center, and Phase 3
            // fractal leaves use it as the world-space cluster origin).
            // The cluster AABB is in world space (host already widened
            // it by params[1] in LeafLocalAabb), so we feed the world
            // AABB straight through.
            if (traceSdfFractal(root.shape, root.params, root.center,
                                c.aabb_min, c.aabb_max,
                                ro, rd, t_min, h.t,
                                iter_budget, epsilon, fractal_de_eps_s,
                                fractal_power, fractal_iters,
                                t, n)) {
                if (t < h.t) {
                    h.hit       = true;
                    h.t         = t;
                    h.normal    = n;
                    h.mat       = c.material;
                    h.albedo    = c.albedo;
                    h.roughness = c.roughness;
                    h.ior       = c.ior;
                    // SDF doesn't source emission; clear so a previous
                    // analytic-prim hit's emission doesn't leak through
                    // when this fractal hit lands closer.
                    h.emission  = float3(0.0);
                    // Editor backend (agent-19): SDF clusters aren't on
                    // the analytic-prim outline path; reset the id so
                    // editor_params.x can't accidentally light up a
                    // fractal hit that landed closer than a selected
                    // sphere.
                    h.prim_id   = 0u;
                }
            }
            continue;
        }
        // --- end SDF Phase 3 -----------------------------------------------
        // t_min is the caller's scale-relative near cut (#256; it used to
        // be an absolute 1e-3 here). t_max = h.t bounds the trace to the
        // current closest hit so an SDF cluster behind an existing closer
        // hit short-circuits.
        //
        // `epsilon` (r_sdf_epsilon) stays ABSOLUTE and is deliberately not
        // converted to footprint units in this PR: it is the sphere-trace
        // SURFACE TERMINATOR, not a self-intersection epsilon, and making
        // it footprint-indexed means evaluating the cone at the current t
        // INSIDE sdfClusterTrace's loop -- a change to SdfPrimitives.slang,
        // which this change does not own.  #253 already names that as a
        // standalone SDF fix rather than part of this phase.
        if (sdfClusterTrace(c, ro, rd, t_min, h.t, iter_budget, epsilon,
                            normal_mode, t, n)) {
            if (t < h.t) {
                h.hit       = true;
                h.t         = t;
                h.normal    = n;
                h.mat       = c.material;
                h.albedo    = c.albedo;
                h.roughness = c.roughness;
                h.ior       = c.ior;
                // SDF doesn't source emission -- see fractal branch above.
                h.emission  = float3(0.0);
                // Editor backend (agent-19): see fractal-branch comment.
                h.prim_id   = 0u;
            }
        }
    }
}
// --- end SDF Phase 1 -------------------------------------------------------

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

// `cone` is the ray cone at THIS ray's origin (issue #256).  It is the
// only new input: the near cut every intersector needs is a pure function
// of it and of the origin, so traceScene derives `t_min` once here rather
// than making every caller carry a second parameter that could drift out
// of step with the first.
HitInfo traceScene(float3 ro, float3 rd, float shutter_t01, PtRayCone cone) {
    float t_min = ptRayTMin(ro, cone);
    HitInfo h;
    h.hit = false; h.t = 1e30;
    h.normal = float3(0,1,0); h.mat = MAT_LAMBERT;
    h.albedo = float3(0.5); h.roughness = 0.0; h.ior = 1.0;
    // Default emission is zero (non-emissive). testAnalyticPrim
    // overwrites this from v4.rgb (#181 polish slot) when a closer
    // prim hit lands; non-analytic hit paths (mesh, SDF, fractal)
    // explicitly reset to zero on overwrite since they don't source
    // emission. Initialising here avoids a stale-emission leak from
    // a previous frame's HitInfo -- HitInfo is stack-local but the
    // field MUST start at 0 so a miss + later overwrite by a
    // non-emissive hit doesn't keep a stale glow.
    h.emission = float3(0.0);
    // Editor backend (agent-19): start with no prim_id; analytic-prim
    // hits overwrite this from v2.w, non-analytic hit paths leave it
    // at 0 so the outline gate's id-match check never accidentally
    // matches a SDF / mesh hit when editor_params.x is non-zero.
    h.prim_id = 0u;
    // Wave 8 PBR (#26): default to no UV and no texture tiles. Only the
    // analytic-prim + mesh hit paths populate these; the SDF / fractal /
    // miss paths leave them at the defaults so the shading-side texture
    // sample is skipped (every tile == kPbrNoTexTile) and those surfaces
    // render with their flat albedo exactly as before.
    h.uv = float2(0.0, 0.0);
    h.tex_tiles = uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    // Ray-cone mip LOD: default to base level (uv_scale 0 -> no LOD). The
    // analytic sphere / plane branches set uv_scale; the trace loop fills
    // tex_lod from the cone footprint before applyPbrTextures.
    h.uv_scale = 0.0;
    h.tex_lod  = 0.0;
    // Wave 9 advanced materials: default to no extension lobe. resolveMatExt
    // (called once at the hit site, after applyPbrTextures) fills these from
    // the push-constant mat_overrides array when the hit's prim id matches
    // an override slot. mat_ext == 0 leaves every advanced lobe inert so the
    // base material renders bit-for-bit.
    h.mat_ext           = 0u;
    h.aniso_amount      = 0.0;
    h.aniso_rotation    = 0.0;
    h.clearcoat_weight  = 0.0;
    h.clearcoat_rough   = 0.03;
    h.subsurface_radius = 0.005;
    h.subsurface_color  = float3(0.9, 0.7, 0.6);

    // Analytic primitives. Each occupies 7 consecutive float4s in
    // `primitives` after #85 (motion blur), #181 polish (emission),
    // #206 (orient), and Wave 8 #26 (PBR texture tiles):
    //   v0 = (xyz: center (current frame) / plane normal,  w: radius / d)
    //   v1 = (rgb: albedo,           a: roughness)
    //   v2 = (x: type,  y: material, z: ior, w: pad)
    //   v3 = (xyz: prev_pos_or_n (sphere center at previous frame), w: pad)
    //   v4 = (xyz: emission (W/sr per channel),                     w: pad)
    //
    // The host partitions the buffer at upload time so the first
    // `linear_prim_count` entries are infinite-extent prims (planes)
    // that must always be linear-tested -- they can't fit in a finite
    // AABB. The remainder is either linear-scanned (when bvh_params.y
    // == 0, i.e. count below r_analytic_bvh_threshold) or traversed
    // via the BVH whose flat node array lives in `bvh_nodes`.
    //
    // shutter_t01 is the per-ray shutter sample in [0, 1] -- 0 = beginning
    // of shutter window (lerp factor 0 -> sphere at prev_pos), 1 = end
    // of shutter (sphere at curr_pos). The host forces shutter_t01 = 1.0
    // when motion blur is off so sphere centers collapse to v0.xyz
    // (bit-identical to pre-#85 path).
    uint linear_prim_count = bvh_params.x;
    uint bvh_node_count    = bvh_params.y;
    // Linear pass over the infinite-extent segment first.
    for (uint i = 0u; i < linear_prim_count && i < prim_count; ++i) {
        testAnalyticPrim(i, ro, rd, shutter_t01, t_min, h);
    }

    if (bvh_node_count == 0u) {
        // BVH inactive: linear-scan the rest of the buffer too.
        for (uint i = linear_prim_count; i < prim_count; ++i) {
            testAnalyticPrim(i, ro, rd, shutter_t01, t_min, h);
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
                    testAnalyticPrim(base + k, ro, rd, shutter_t01, t_min, h);
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

    // --- SDF Phase 1 (#97) ---------------------------------------------
    // SDF clusters. Each cluster is a self-contained op-tree of
    // analytic SDFs (sphere/box/torus/capsule/rounded box) blended via
    // smooth-CSG ops. The shape's hit-record format is identical to
    // the analytic-prim path (pos + normal + material id) so the BSDF
    // / NEE / denoiser downstream see no difference. Linear scan over
    // all clusters; each cluster does its own slab-test against its
    // tight world-space AABB before launching the sphere-trace, so
    // the heavy work is only paid when the ray actually enters a
    // cluster's footprint.
    traceSdfClusters(ro, rd, t_min, h);
    // --- end SDF Phase 1 -----------------------------------------------

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
    //
    // Mesh motion blur (wave-7 #21, follow-up to PR #85): the baked
    // mesh sits in its untransformed "rest" frame (world origin for
    // the canonical CSG demo bake). The engine carries two world-
    // space translations -- prev (last frame) and curr (this frame)
    // -- and the shader computes per-ray effective translation
    //   effective(t01) = lerp(prev, curr, t01)
    // at shutter sample t01 ∈ [0, 1]. Translating the mesh by
    // +effective(t01) is mathematically equivalent to SHIFTING THE
    // RAY ORIGIN by -effective(t01) before intersection against the
    // static baked mesh. When r_motion_blur is off the shader forces
    // t01 = 1.0 (PR #85 plumbing in the per-sample loop) so the lerp
    // collapses to curr -- the mesh renders at its current world
    // position with zero streak.
    //
    // The .w gate on mesh_motion_prev shortcuts the entire shift
    // when the mesh has neither moved nor been translated -- the
    // common static-scene case. Both the HW RayQuery and SW BVH
    // paths use the same shifted origin so the streak is consistent
    // across backends. The TLAS instance transform stays identity --
    // no per-frame BLAS / TLAS rebuild is required. Translation
    // preserves normals so q.CommittedObjectToWorld4x3() returning
    // identity is still correct downstream.
    float3 mesh_shift = (mesh_motion_prev.w > 0.5)
                            ? lerp(mesh_motion_prev.xyz, mesh_motion_curr.xyz, shutter_t01)
                            : float3(0.0, 0.0, 0.0);
    float3 ro_mesh = ro - mesh_shift;
#if !defined(PT_TARGET_SPIRV) || !defined(PT_SPIRV_NO_RAYQUERY)
    if (tlas_present != 0u) {
        RayDesc r;
        r.Origin    = ro_mesh;
        r.Direction = rd;
        r.TMin      = t_min;
        r.TMax      = h.t;     // limit search to current closest

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, r);
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float t_mesh = q.CommittedRayT();
            if (t_mesh < h.t) {
                uint   prim = q.CommittedPrimitiveIndex();
                float2 bary = q.CommittedTriangleBarycentrics();
                {
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
                // Mesh path doesn't source emission -- clear so a prior
                // analytic-prim hit's emission doesn't leak through when
                // this mesh hit is closer.
                h.emission = float3(0.0);
                // Editor backend (agent-19): mesh hits aren't in the
                // analytic-prim outline scope; clear the id so a
                // selected sphere can't "shine through" a closer mesh
                // hit because the stale prim_id was left from the
                // analytic prim test that ran first.
                h.prim_id = 0u;
                // Wave 8 PBR (#26): barycentric-interpolate the per-vertex
                // UVs (mesh_uvs parallel to mesh_positions) and carry the
                // mesh's material texture tiles. Bary from the HW query is
                // (b1, b2) for v1/v2; v0 weight = 1 - b1 - b2. When the
                // mesh has no UV buffer the host forces the tile indices
                // to kPbrNoTexTile so the sample is skipped (h.uv stays 0).
                float2 uv0 = float2(mesh_uvs[i0 * 2 + 0], mesh_uvs[i0 * 2 + 1]);
                float2 uv1 = float2(mesh_uvs[i1 * 2 + 0], mesh_uvs[i1 * 2 + 1]);
                float2 uv2 = float2(mesh_uvs[i2 * 2 + 0], mesh_uvs[i2 * 2 + 1]);
                h.uv = uv0 * (1.0 - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
                h.tex_tiles = mesh_tex_indices;
                }
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
        // Mesh motion blur (wave-7 #21): we walk the BVH with
        // ro_mesh = ro - mesh_shift(t01) instead of ro, which puts the
        // intersection in the static-mesh frame. The AABB-slab cull,
        // Möller-Trumbore intersect, and returned `t_tri` are all in
        // that frame -- but because the static mesh is just a
        // translation of the moving mesh, the same ray parameter `t`
        // satisfies (ro + t*rd) on the moving mesh. So the host
        // doesn't need to know whether the hit came from the HW or SW
        // path; both return the same `t`.
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
                if (!intersectAabb(ro_mesh, rd_inv, node.aabb_min, node.aabb_max, t_mesh)) continue;
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
                        float3 tvec = ro_mesh - v0;
                        float u = dot(tvec, pv) * inv_det;
                        if (u < 0.0 || u > 1.0) continue;
                        float3 qv = cross(tvec, e1);
                        float v = dot(rd, qv) * inv_det;
                        if (v < 0.0 || u + v > 1.0) continue;
                        float t_tri = dot(e2, qv) * inv_det;
                        if (t_tri > t_min && t_tri < t_mesh) {
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
            // SW mesh path doesn't source emission either -- clear for
            // the same reason as the HW path above.
            h.emission = float3(0.0);
            // Editor backend (agent-19): same id-clear as the HW mesh
            // branch above; mesh hits aren't in the outline scope.
            h.prim_id = 0u;
            // Wave 8 PBR (#26): the closest-hit loop discarded its
            // Möller-Trumbore (u, v), so re-derive barycentrics at the
            // hit point in the shifted (static-mesh) frame -- the same
            // frame v0/v1/v2 live in. Cramer's-rule projection onto the
            // triangle's edge basis. Then interpolate the per-vertex UVs.
            float3 hp_mesh = ro_mesh + rd * t_mesh;
            float3 e1b = v1 - v0;
            float3 e2b = v2 - v0;
            float3 dp  = hp_mesh - v0;
            float d00 = dot(e1b, e1b);
            float d01 = dot(e1b, e2b);
            float d11 = dot(e2b, e2b);
            float d20 = dot(dp, e1b);
            float d21 = dot(dp, e2b);
            float denom = d00 * d11 - d01 * d01;
            float bw1 = 0.0, bw2 = 0.0;
            if (abs(denom) > 1e-20) {
                bw1 = (d11 * d20 - d01 * d21) / denom;   // weight of v1
                bw2 = (d00 * d21 - d01 * d20) / denom;   // weight of v2
            }
            float2 uv0 = float2(mesh_uvs[i0 * 2u + 0u], mesh_uvs[i0 * 2u + 1u]);
            float2 uv1 = float2(mesh_uvs[i1 * 2u + 0u], mesh_uvs[i1 * 2u + 1u]);
            float2 uv2 = float2(mesh_uvs[i2 * 2u + 0u], mesh_uvs[i2 * 2u + 1u]);
            h.uv = uv0 * (1.0 - bw1 - bw2) + uv1 * bw1 + uv2 * bw2;
            h.tex_tiles = mesh_tex_indices;
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
// by the third pass). With the celestials composite enabled the
// primary-miss code subtracts this value from the sky term so the
// denoiser never sees it; the engine's StarsComposite kernel re-adds
// it post-denoise with N aperture-sampled rays per pixel
// (StarsComposite.slang carries a verbatim copy of this function so
// any change must mirror there too). The helper is also called from
// procSky() for the legacy (star-split off / denoiser off) and
// secondary-bounce paths, so stars still contribute to indirect
// lighting in the scene.
float3 starsOnly(float3 rd, float3 sun) {
    // Planetary P0 (#254): the observer's local up, not world +Y. Sky-dome
    // helpers are evaluated in the CAMERA's frame -- see procSky()'s
    // precondition note for why the secondary-bounce approximation is
    // acceptable. Planar frame -> exactly rd.y / sun.y, bit-for-bit.
    float3 up      = localUp(pos_fovtan.xyz);
    float cosTheta = clamp(dot(rd, up), -1.0, 1.0);
    float sun_elev = dot(up, sun);
    float day      = smoothstep(-0.10, 0.20, sun_elev);
    // Planetary P6 (#260): WHERE THE STARS STOP.
    //
    // `cosTheta > -0.05` is a flat-Earth statement -- "anything more than
    // 2.9 degrees below the observer's horizontal is ground". At eye
    // height that is nearly true (the visible horizon dips 3.7 arcmin)
    // and the 2.9 degrees is slack. From 400 km the hard limb sits 19.8
    // degrees below local horizontal, so the band from 2.9 to 19.8
    // degrees down is genuine black sky full of stars, and the gate
    // paints a hard horizontal line across the frame with stars above it
    // and nothing below -- a discontinuity that opens up as the camera
    // climbs, which is exactly what this phase exists to close.
    //
    // The physically correct criterion is whether the body is in the way,
    // and it needs no threshold at all. Only available when there IS a
    // body: without one there is no sphere to test and the 2.9-degree
    // slack stays, so every pre-#260 fixture is bit-stable (none of them
    // sets a planetary frame and stars together).
    bool star_visible = (cosTheta > -0.05);
    if (planet_center_radius.w > 0.0) {
        float b0, b1;
        star_visible = !(ptSphereRoots(pos_fovtan.xyz, rd,
                                       planet_center_radius.xyz,
                                       planet_center_radius.w, b0, b1)
                         && b0 > 0.0);
    }
    // BSC starmap (real Yale Bright Star Catalog, J2000 frame). Sampled
    // additively whenever the host has uploaded the texture and we're
    // looking above the horizon. Falls back to the procedural cell-hash
    // scatter below when the starmap isn't available -- the host gates
    // exposure_pad.z (BSC) and .y (procedural) to be mutually exclusive.
    if (star_visible && exposure_pad.z > 0.5 && day < 0.6) {
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
        // averaged flat to the eye.) +/-45%% amplitude over a per-
        // star 1-3 Hz base frequency, scaled by exposure_pad.w.
        // exposure_pad.w doubles as the on/off flag (0 = twinkle off
        // -> skip the modulation) AND as the user-tunable speed
        // multiplier (r_stars_twinkle_speed, default 0.3 -- calm
        // night cadence around 0.3-0.9 Hz / period ~1-3 s; higher
        // values drive toward more nervous scintillation).
        if (exposure_pad.w > 0.001) {
            int3 q  = int3(floor(j * 38.0));   // ~1.5 deg bins
            uint qh = uint(q.x + 128) * 65536u
                    + uint(q.y + 128) * 256u
                    + uint(q.z + 128);
            uint h    = pcgHash(qh + 0xb7e1u);
            // Two-component scintillation: a single sine looks
            // metronomic on sharp pixel stars (especially obvious
            // now that the stateless composite doesn't smear them
            // through the denoiser). Sum two sines at incommensurate
            // frequencies -- a dominant low-frequency beat plus a
            // smaller high-frequency wobble -- and the combined
            // waveform never quite repeats, mimicking the
            // quasi-aperiodic look of real atmospheric scintillation
            // while still feeling rhythmic.
            //
            // Component 1: 1-3 Hz base * exposure_pad.w speed,
            //              amplitude 0.30 (the dominant rhythm).
            // Component 2: 2.7-5.3x component 1's frequency,
            //              amplitude 0.15 (the random-feeling
            //              flicker on top).
            // Total peak swing 0.45 around mean 1.0, matching the
            // previous single-sine amplitude.
            float ph  = float(h & 0xFFFFu) * (1.0 / 65536.0);
            float fhz_base = 1.0 + float((h >> 16) & 0xFFu) * (2.0 / 256.0);
            uint h2   = pcgHash(h + 0xc0ffeeu);
            float ph2 = float(h2 & 0xFFFFu) * (1.0 / 65536.0);
            float fhz_mult = 2.7 + float((h2 >> 16) & 0xFFu) * (2.6 / 256.0);
            float speed = exposure_pad.w;
            float fhz1 = fhz_base * speed;
            float fhz2 = fhz_base * fhz_mult * speed;
            float t   = float(frame_index) * (1.0 / 60.0);
            float w   = 1.0
                      + 0.30 * sin(6.28318530718 * (ph  + t * fhz1))
                      + 0.15 * sin(6.28318530718 * (ph2 + t * fhz2));
            stars *= w;
        }
        // Fade with day so stars vanish at sunrise.
        return stars * (1.0 - day);
    } else if (star_visible && exposure_pad.y > 0.5 && day < 0.6) {
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
    // Mode 4 (physical, #260) composites its celestials inside skyColor
    // and the engine leaves the post-denoise composite off for it, so
    // this helper -- which exists only to let the composite SUBTRACT the
    // inline stars -- must not claim mode 4's stars.
    if (mode == 2u) {
        return starsOnly(rd, sun_and_mode.xyz);
    }
    return float3(0.0);
}

// procSky's inline sun disk + halo, factored out so the b==0
// celestials-composite subtraction can peel them off denoise_color.
// procSky() ALSO renders an inline sun disk/halo (lines below) on top
// of the regular `sunDisc()` overlay that skyColor() adds afterwards;
// without subtracting THIS too, the denoiser still sees a sharp solar
// disk and StarsComposite paints another sun on top -- two suns, one
// noisy through SVGF/MetalFX, one clean from the composite. The
// `wide` sunset-scatter term is INTENTIONALLY excluded: it's a smooth
// atmospheric gradient (sunset glow across the sky) that benefits from
// being denoised, isn't localized at the sun direction the same way,
// and the composite has no equivalent to add back.
//
// Mirrored verbatim in shaders/StarsComposite.slang so any tuning here
// must mirror there too (or the composite will under/over-restore the
// sun and the disc brightness will drift between
// composite-on / composite-off A/B).
float3 procSkyInlineSunDiscAndHalo(float3 rd, float3 sun) {
    float cosGamma = clamp(dot(rd, sun), -1.0, 1.0);
    // Planetary P0 (#254): solar elevation at the observer, not the
    // global sun.y. Planar frame -> sun.y bit-for-bit.
    float sun_elev = dot(localUp(pos_fovtan.xyz), sun);
    float day      = smoothstep(-0.10, 0.20, sun_elev);
    float disk     = smoothstep(0.99996, 0.99999, cosGamma);
    float3 sun_color_high = float3(1.00, 0.95, 0.85);
    float3 sun_color_low  = float3(1.00, 0.55, 0.20);
    float3 sun_col = lerp(sun_color_low, sun_color_high, day);
    float3 result  = disk * 50.0 * sun_col * day;
    float halo     = pow(max(0.0, cosGamma), 80.0);
    result += halo * sun_col * 0.30 * day;
    return result;
}

// --- Wave 9 hosek-sky -------------------------------------------------------
// Hosek-Wilkie (2012) physically-based analytic sky-dome radiance.
//
// This is the REAL Hosek-Wilkie model, driven by the reference
// ArHosekSkyModel RGB coefficient dataset (Hosek & Wilkie, "An Analytic
// Model for Full Spectral Sky-Dome Radiance", SIGGRAPH 2012; reference
// code v1.4a). The dataset's per-turbidity/albedo/elevation "cook" step --
// which turns the 3 uniforms (turbidity, ground albedo, solar elevation)
// into the 9 shape coefficients A..I plus a radiance magnitude L_M per RGB
// channel -- runs ONCE per frame on the CPU (see src/engine/HosekSkyModel.h
// and Engine.cpp), and the cooked coefficients arrive here in the push
// constant (hosek_cfg / hosek_radiance). This shader evaluates only the
// cheap per-view-direction shape function, exactly the reference
// architecture (cook once, evaluate per direction).
//
// [History: this function previously fabricated the ~130 coefficients as
// hand-authored round numbers falsely attributed to a "Helmer fit". Those
// tables collapsed the blue channel at the zenith (dark olive sky, blue
// clamping to 0 at high turbidity). They are gone; every number now traces
// to the reference dataset.]
//
// Radiance model (per channel c), evaluated below from the cooked coeffs:
//   L(theta, gamma) = F_c(theta, gamma) * Z_c
//   F(theta, gamma) = (1 + A*exp(B / (cos(theta) + 0.01)))
//                   * (C + D*exp(E*gamma) + F*cos^2(gamma)
//                      + G*chi(H, gamma) + I*sqrt(cos(theta)))
//   chi(g, a) = (1 + cos^2 a) / (1 + g^2 - 2 g cos a)^1.5
// where theta is the angle from zenith, gamma the angle from the sun,
// and Z_c the channel radiance magnitude (the model's "L_M").

// chi term -- Mie-style forward-scatter lobe in the radiance shape.
float3 hosekChi(float3 g, float cos_gamma) {
    float3 num = 1.0 + cos_gamma * cos_gamma;
    float3 den = 1.0 + g*g - 2.0 * g * cos_gamma;
    return num / pow(max(den, float3(1e-4)), float3(1.5));
}

// (The per-frame "cook" that turns turbidity/albedo/elevation into the 9
// coefficients A..I + radiance L_M lives on the CPU now --
// src/engine/HosekSkyModel.h::Cook, called from Engine.cpp -- and the
// cooked values arrive in hosek_cfg[] / hosek_radiance in the push. This
// used to be a fabricated hand-tuned table; see the header note above.)

// Hosek-Wilkie sky radiance for view direction rd given the sun
// direction. Mirrors procSky's contract: returns HDR linear radiance,
// callers add the sun disc / stars on top. Used for BOTH the visible
// sky-dome (primary miss) and env lighting (skyColor at secondary
// bounces), so the path tracer's ambient term is consistent with what
// the camera sees.
float3 hosekSky(float3 rd, float3 sun) {
    // theta = angle from zenith, gamma = angle from the sun. Planetary P0
    // (#254): zenith is the observer's LOCAL up, so the Hosek dome tilts
    // with the surface normal of the planet instead of staying pinned to
    // world +Y. Planar frame -> exactly rd.y, bit-for-bit.
    float3 up       = localUp(pos_fovtan.xyz);
    float cos_up    = dot(rd, up);
    float cos_theta = clamp(cos_up, 0.0, 1.0);    // below-horizon rays clamp to horizon
    float cos_gamma = clamp(dot(rd, sun), -1.0, 1.0);
    float gamma     = acos(cos_gamma);

    // Cooked coefficients (real Hosek-Wilkie RGB dataset), computed on the
    // CPU this frame and delivered in the push. Reference index order:
    //   [0]=A [1]=B [2]=C [3]=D [4]=E [5]=F [6]=G [7]=I(sqrt) [8]=H(Mie g).
    float3 cA = hosek_cfg[0].xyz;
    float3 cB = hosek_cfg[1].xyz;
    float3 cC = hosek_cfg[2].xyz;
    float3 cD = hosek_cfg[3].xyz;
    float3 cE = hosek_cfg[4].xyz;
    float3 cF = hosek_cfg[5].xyz;
    float3 cG = hosek_cfg[6].xyz;
    float3 cI = hosek_cfg[7].xyz;   // sqrt(cos theta) zenith coefficient
    float3 cH = hosek_cfg[8].xyz;   // Mie asymmetry g
    float3 radZ = hosek_radiance.xyz;

    // Radiance shape function F(theta, gamma), evaluated per channel.
    float3 expM = float3(1.0) + cA * exp(cB / (cos_theta + 0.01));
    float3 chi  = hosekChi(cH, cos_gamma);
    float3 expG = exp(cE * gamma);
    float  cg2  = cos_gamma * cos_gamma;
    float  sct  = sqrt(cos_theta);
    float3 lobe = cC + cD * expG + cF * cg2 + cG * chi + cI * sct;
    float3 F    = expM * lobe;

    // Below the horizon the analytic F can ring negative or blow up;
    // fade the model out smoothly just under the horizon so reflective
    // surfaces and downward bounces don't pick up garbage.
    float horizon_fade = smoothstep(-0.05, 0.02, cos_up);

    float3 sky = max(F, float3(0.0)) * radZ * horizon_fade;

    // Night: when the sun is well below the horizon there is no Hosek
    // daylight; fall to a deep navy and let stars carry the sky. Reuse
    // the procedural day-blend band so the hosek/procedural night look
    // matches and the celestial overlays read identically.
    float day = smoothstep(-0.10, 0.20, dot(up, sun));
    float3 night = float3(0.005, 0.010, 0.030);
    sky = lerp(night, sky, day);

    // Stars (BSC / procedural fallback), matching procSky so night skies
    // and the celestials composite behave the same across modes.
    sky += starsOnly(rd, sun);

    return max(sky, float3(0.0));
}
// --- end Wave 9 hosek-sky ---------------------------------------------------

// procSky -- procedural sky gradient + sun halo.
//
// PRECONDITION / approximation: the horizon-dip rebias is driven by
// the CAMERA altitude (push_constants.pos_fovtan.y), not the per-call
// ray origin. procSky() is invoked from primary camera rays AND from
// secondary bounces (mirror reflections, NEE sky lookups deep inside
// traceScene). For a high-altitude camera, a reflective puddle on the
// ground will sample procSky() with rd.y > 0 and still see the 10 km
// dipped horizon as if the puddle itself were at 10 km. The visible
// effect is small (the puddle reflects mostly upper hemisphere, where
// the dip has zero visual impact -- the bias only shifts the painted
// horizon LINE downward) and the alternative (plumbing per-ray ro
// through every traceScene call) is a much larger surgery for what
// is essentially a procedural-mode debug visual. If/when we move the
// sky to a per-hit transmittance LUT, this approximation goes away.
float3 procSky(float3 rd, float3 sun) {
    // Planetary P0 (#254): zenith and solar elevation are measured
    // against the observer's LOCAL up, not world +Y. In the planar frame
    // localUp() is exactly float3(0,1,0), so cosTheta_raw / sun_elev are
    // rd.y / sun.y bit-for-bit and this function is unchanged.
    float3 up = localUp(pos_fovtan.xyz);
    float cosTheta_raw = clamp(dot(rd, up), -1.0, 1.0);  // 1=zenith, 0=mathematical-horizon, -1=nadir
    // (#280 removed `cosGamma` and `sun_elev` from this scope: every term
    // that read them -- the day/night blend, the sunset cross-fade and the
    // painted wide forward-scatter lobe -- is now either integrated by the
    // host cook or marched as real aerosol.  The solar geometry the sky
    // still needs is inside the cook, where it is a ray direction rather
    // than a smoothstep edge.)

    // Curved-Earth visible-horizon dip (issue #51). On a flat Earth the
    // geometric horizon sits at rd.y = 0. On a spherical Earth of radius
    // R, a camera at altitude h above the surface sees the horizon dip
    // below local-up by acos(R/(R+h)) -- that is, the lowest sky ray
    // that grazes the surface has
    //     rd.y_horizon = -sqrt(2*h/R + (h/R)^2) ≈ -sqrt(2*h/R)
    // for small h/R. At eye height 1.7 m the dip is ~3.7 arcmin (invisible).
    // At a 1 km mountain it's ~1.0 deg. At a 10 km airliner it's ~3.2 deg.
    // At a 100 km space-edge view it's ~10 deg.
    //
    // We rebias the gradient so the visible horizon-line in the painted
    // sky sits at the geometrically-correct dipped angle instead of at
    // rd.y = 0. This is the procedural-sky side of the "distant clouds
    // visually merge with the horizon" requirement -- without the dip,
    // a high-altitude camera shows a flat-Earth pink-on-blue band cutting
    // across the lower screen, even though the real visible horizon
    // would be below that.
    //
    // sun_extra.w packs the planet radius in metres; 0 means "planar
    // mode" (issue #51 debug toggle) so the dip is exactly 0 and this
    // collapses to the legacy hand-tuned gradient. We use the camera
    // altitude inferred from the ray origin -- in this function we
    // only have rd, but the engine's ground plane is at world y=0 so
    // the camera altitude is the path-tracer's ro.y. We don't have ro
    // here directly, but the sky-color call sites use camera-origin
    // rays so we can pull cam_pos.y from the push constants. We avoid
    // a global by recomputing the dip in callers that have ro -- here
    // we expose a helper. For zero camera height (the common case),
    // dipping is a no-op so the change is invisible.
    //
    // Below: shift the gradient so cosTheta=0 hits the dipped horizon
    // line. The camera's altitude comes from planetAltitude(), which is
    // pos_fovtan.y in the planar frame (#254).
    float cam_h = max(planetAltitude(pos_fovtan.xyz), 0.0);
    float planet_R = sun_extra.w;
    float horizon_dip_sin = 0.0;
    if (planet_R > 0.0 && cam_h > 0.0) {
        float ratio = cam_h / planet_R;
        // sin(dip) = sqrt(2h/R + (h/R)^2). Well-defined and monotonic all
        // the way to h = R (the sphere subtends a full hemisphere), and
        // the clamp below keeps the result a legal sine. #254 removed an
        // earlier min(..., 0.5) cap here: it pinned the dip at ~30 deg,
        // but the true dip at a 400 km orbit is 70.2 deg, so the cap
        // painted a flat-Earth horizon band across every orbital view.
        // It never fired below ~797 km of altitude, so no existing scene
        // is affected.
        horizon_dip_sin = min(sqrt(2.0 * ratio + ratio * ratio), 1.0);
    }
    // Shift cosTheta upward by the dip so the painted horizon line moves
    // down on screen to match the visible horizon. Clamp to [-1,1] so
    // the smoothsteps below stay bounded.
    float cosTheta = clamp(cosTheta_raw + horizon_dip_sin, -1.0, 1.0);

    // --- #280: the palette is integrated, not painted ---------------------
    //
    // Until #280 the three lines below were six hand-authored literals --
    // zenith_day (0.32, 0.55, 0.92), horizon_day (0.78, 0.88, 1.00), a
    // sunset pair, and a `sunset` smoothstep to cross-fade between them --
    // in units where the sun was 80.  They are now three RADIANCES that the
    // host integrates once per frame through this scene's actual medium
    // (src/renderer/Atmosphere.cpp CookSky): Rayleigh single scattering,
    // ozone absorption on the sun's slant path, and Hillaire 2020's
    // multiple-scattering term, at the zenith and at the horizon in the
    // sun's azimuth and the anti-solar one.
    //
    // What survives from the painted version is the SHAPE -- the
    // smoothstep in cosTheta below, and side_factor -- and that is exactly
    // what `procedural` is for: a cheap dome interpolating a physical
    // palette, against `physical` which marches every pixel.  The sunset
    // cross-fade is gone because the physics does it: at 2 deg solar
    // elevation the cooked sun-side horizon is (10.10, 6.22, 2.57) W/m^2/sr
    // -- orange, from the medium, with nothing authored.
    //
    // side_factor's "horizontal" projection is onto the LOCAL horizon plane
    // (#254), i.e. the component perpendicular to local up.
    float3 rd_h        = rd  - up * dot(rd,  up);
    float3 sun_h       = sun - up * dot(sun, up);
    float side_factor  = smoothstep(0.0, 0.7, dot(normalize(rd_h),
                                                  normalize(sun_h)));
    float3 horizon_col = lerp(sky_cook_horizon_anti.rgb,
                              sky_cook_horizon_sun.rgb, side_factor);
    float3 zenith_col  = sky_cook_zenith.rgb;

    // Vertical gradient from horizon to zenith.
    float t = smoothstep(0.0, 0.40, cosTheta);
    float3 sky = lerp(horizon_col, zenith_col, t);

    // Night floor, ADDED rather than cross-faded (#280).  The cooked
    // palette goes to zero on its own when the sun is far enough below the
    // horizon -- that is what the integral does -- so the `day` smoothstep
    // that used to blend to night has nothing left to do and is gone.
    // What remains is a floor standing in for airglow and integrated
    // starlight, which this engine does not model and which really is
    // additive: at noon it is 0.028 against a zenith of ~15, and at
    // midnight it is the whole sky.  Carried across the unit rebase by the
    // one documented factor, so its brightness relative to the physical
    // sky is where its author put it.
    sky += float3(0.005, 0.010, 0.030) * kPtLegacySkyScale;

    // Star contribution. Routed through starsOnly() so the same logic
    // is reusable from the primary-miss celestials-composite path
    // (issue #46). When the composite is active, the b==0 site
    // subtracts this term from the sky so the denoiser doesn't see
    // it; secondary bounces and NEE still call into here through
    // skyColor() so stars contribute to indirect lighting.
    sky += starsOnly(rd, sun);

    // Sun disk + soft halo. Factored into procSkyInlineSunDiscAndHalo
    // so the celestials-composite subtraction at b==0 can mirror these
    // exact terms when removing the inline sun from denoise_color.
    sky += procSkyInlineSunDiscAndHalo(rd, sun);

    // #280 REMOVED a "wider, dimmer secondary scatter" term here:
    //
    //     sky += pow(max(0, cosGamma), 6) * sun_col * sunset * 0.50;
    //
    // a painted forward-scattering lobe that spread sunset orange across
    // the sky, driven by the same `sunset` smoothstep the palette above no
    // longer has.  It is a deletion rather than a loss, and for the same
    // reason #260 deleted the painted sun halo from mode 4: that broad
    // glow around a low sun IS forward Mie scattering, and the aerosol
    // half of the in-scatter march computes it through Cornette-Shanks at
    // the real aerosol loading.  Painting it as well would double-count
    // it, which is the pattern this whole issue exists to remove.
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
    // Planetary P0 (#254): every "how high is the sun" term below is
    // measured against the observer's local up rather than world +Y.
    // Planar frame -> world_up is exactly float3(0,1,0) and sun_elev is
    // exactly sun_dir.y, so the disc is unchanged bit-for-bit.
    float3 world_up = localUp(pos_fovtan.xyz);
    float  sun_elev = dot(world_up, sun_dir);
    if (sun_elev < -0.10) return float3(0.0);
    float ang = dot(rd, sun_dir);
    // r_sun_size * astronomical-distance ratio. perihelion ~1.017
    // (sun looks bigger), aphelion ~0.984. Half-angle scales linearly.
    float sun_mult = max(sun_extra.x * sun_extra.y, 0.1);
    float r_deg    = 0.55 * sun_mult;
    float kCosR    = cos(r_deg * 0.0174533);
    // Halo extent grows from cos(2deg) at zenith to cos(8deg) near
    // horizon. The horizon halo is what you see as the "bigger" sun
    // at sunset -- it's atmospheric scattering, not the disc itself.
    float halo_deg   = lerp(2.0, 8.0, saturate(1.0 - sun_elev));
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
    float am  = 1.0 / max(sun_elev, 0.04);
    float atm = exp(-0.30 * am);
    float warmth = saturate((am - 1.0) * 0.18);
    float3 col = lerp(float3(1.00, 0.98, 0.94),
                      float3(1.55, 0.55, 0.20), warmth);

    // Halo: soft bloom around the disc.
    float halo = 0.0;
    if (ang > kCosHalo) {
        float t_h = (ang - kCosHalo) / (1.0 - kCosHalo);
        // Steeper falloff at zenith (tight halo), wider at horizon.
        float falloff = lerp(8.0, 3.0, saturate(1.0 - sun_elev));
        halo = pow(t_h, falloff);
    }

    // r_sun_disc_brightness disc multiplier (sun_extra2.x) -- the daytime mirror
    // of moonDisc's moon_extra.z knob. Disc and halo scale together so the
    // surrounding bloom stays proportionate. Default 1.0 reproduces the
    // historical hardcoded disc intensity bit-for-bit. Scales only the
    // rendered disc here, never the sun_rad NEE / cloud-march lighting.
    float bright = max(sun_extra2.x, 0.0);
    if (inside_disc) {
        // Bright disc -- sun is enormously brighter than anything
        // else; ACES will clip the centre to white but the limb +
        // halo carry the warmth.
        return (col * (250.0 * atm)
             + col * halo * atm * 4.0) * bright;
    }
    return col * halo * atm * 4.0 * bright;
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
    float3 m_dir = mph.xyz;
    // Planetary P0 (#254): lunar elevation at the observer, not the
    // global mph.y. Planar frame -> mph.y bit-for-bit. (The tangent
    // frame below picks an arbitrary basis axis off m_dir.y -- that is a
    // parameterisation choice, not an up-vector, and must NOT move or
    // the moon texture rotates.)
    float moon_elev = dot(localUp(pos_fovtan.xyz), m_dir);
    if (moon_elev < -0.10) return float3(0.0);
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
    float moon_am   = 1.0 / max(moon_elev, 0.04);
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
        // r_moon_disc_brightness (moon_extra.z). 1.0 = baseline; 0.7
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


float3 acesTonemap(float3 c) {
    const float a = 2.51, b = 0.03, d = 2.43, e = 0.59, f = 0.14;
    return saturate((c*(a*c+b))/(c*(d*c+e)+f));
}

// --- Wave 9 tonemap (#27 follow-up) ---------------------------------------
// Modern display-transform operators selectable at runtime via the
// r_tonemap_op cvar (host packs the enum into tonemap_params.x). ACES
// (op 0) stays the engine default and the existing call sites collapse to
// the unchanged acesTonemap() above, so committed goldens are bit-for-bit
// preserved. The AgX + Khronos PBR Neutral curves are duplicated verbatim
// in shaders/Tonemap.slang (the post-denoise Metal pass) -- same reasoning
// as acesTonemap / srgb_oetf: each kernel stays a self-contained TU and any
// drift between the inline (denoiser-off) and post-pass (denoiser-on) copies
// would show as a visible A/B diff, so the constants MUST stay identical.
//
// Operator enum (mirror of pt::engine kTonemapOp* in Engine.cpp):
//   0 = aces      (Narkowicz fit, default)
//   1 = agx       (Troy Sobotka AgX, Minimal/Filament 6th-order fit)
//   2 = khronos_pbr_neutral (Khronos glTF reference)
//   3 = reinhard  (extended Reinhard, white point 1)
//   4 = linear    (clamp only -- debug / pass-through)
static const uint kTonemapAces     = 0u;
static const uint kTonemapAgx      = 1u;
static const uint kTonemapKhronos  = 2u;
static const uint kTonemapReinhard = 3u;
static const uint kTonemapLinear   = 4u;

// AgX -- "Minimal AgX" by Benjamin Wrensch (a compact fit of Troy
// Sobotka's AgX), the same approximation Filament ships. Three stages:
//   1. transform into the AgX working space (a desaturating 3x3 matrix),
//   2. log2 encode over the [-12.47, +4.026] EV range and normalise to
//      [0, 1], then push through a per-channel 6th-order sigmoid polynomial,
//   3. apply the AgX "punchy" look (slope/power/saturation) and the inverse
//      working-space matrix, ending in sRGB-display linear.
// Input is scene-linear (post-exposure). Output is display-linear sRGB; the
// sRGB OETF is applied separately at the swapchain-write site, exactly as
// for ACES. The curve already lands in [0, 1] so no extra saturate() needed.
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
    // AgX inset (working-space) matrix -- rows match the Filament/Wrensch
    // constants. Slang float3x3 is column-major on construction, so build
    // it from explicit rows via transpose of the row layout: we encode each
    // matrix-vector product as dot()s to stay layout-agnostic.
    const float3 ir0 = float3(0.842479062253094,  0.0784335999999992, 0.0792237451477643);
    const float3 ir1 = float3(0.0423282422610123, 0.878468636469772,  0.0791661274605434);
    const float3 ir2 = float3(0.0423756549057051, 0.0784336,          0.879142973793104);
    float3 v = float3(dot(ir0, c), dot(ir1, c), dot(ir2, c));

    // Log2 encode, normalise to [0,1] over the AgX EV window.
    const float kMinEv = -12.47393;
    const float kMaxEv =   4.026069;
    v = clamp(log2(max(v, 1e-10)), kMinEv, kMaxEv);
    v = (v - kMinEv) / (kMaxEv - kMinEv);

    // Sigmoid contrast.
    v = agxDefaultContrastApprox(v);

    // AgX "punchy" look: gamma + per-channel saturation about luma.
    const float3 kLuma = float3(0.2126, 0.7152, 0.0722);
    float luma = dot(v, kLuma);
    const float3 offset = float3(0.0);
    const float3 slope  = float3(1.0);
    const float3 power  = float3(1.0);
    const float  sat    = 1.0;
    v = pow(max(v * slope + offset, float3(0.0)), power);
    v = luma + sat * (v - luma);

    // Inverse (outset) matrix back to sRGB-display linear.
    const float3 or0 = float3( 1.19687900512017,  -0.0980208811401368, -0.0990297440797205);
    const float3 or1 = float3(-0.0528968517574562, 1.15190312990417,   -0.0989611768448433);
    const float3 or2 = float3(-0.0529716355144438,-0.0980434501171241,  1.15107367264116);
    v = float3(dot(or0, v), dot(or1, v), dot(or2, v));
    // Linearize out of AgX's base encoding. The sigmoid fit above
    // outputs in the AgX working encoding (~sRGB 2.2-gamma); the
    // reference (Wrensch's agxEotf, Filament) applies pow(2.2) here.
    // Without it the swapchain's sRGB OETF encodes an ALREADY-encoded
    // signal -- washed-out, lifted-black AgX frames.
    v = pow(max(v, float3(0.0)), float3(2.2));
    return saturate(v);
}

// Khronos PBR Neutral -- the official glTF-Sample-Viewer neutral tone
// mapper (Khronos 3D Formats group). Designed to preserve material albedo
// and hue: it leaves the darks/mids untouched, compresses only the bright
// tail, and desaturates toward white as luminance approaches the ceiling so
// saturated highlights roll off without hue shift. Reference is ~15 lines.
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

// Dispatch on the operator enum. Input is scene-linear (post-exposure);
// output is display-linear (sRGB OETF applied at the call site). op == 0
// returns exactly acesTonemap(c) so the existing default is byte-identical.
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

// ---------- Analytic light primitives NEE (#73) ------------------------
// First-class scene lights: point / spot / sphere area / quad area.
// Lifted out of the HDRI sky so a scene can be lit even with no env
// map (the existing hdri_lights_count > 0 path is sun-on-the-sphere
// only). Phase 1 sampling is the naive uniform single-pick: choose a
// light uniformly at random from the count, sample a point on it,
// trace a shadow ray, accumulate (Le * BRDF * cos / pdf). Variance is
// fine up to ~100 lights -- beyond that the light tree (#129) takes
// over with O(log N) importance selection.
struct LightSample {
    float3 dir;        // unit vector from hit point toward the light sample
    float3 Le;         // radiance reaching the surface along dir (no
                       //   visibility yet -- transmittance() applies that)
    float  pdf;        // probability density of this sample (solid angle).
                       //   Selection (1/N) is folded in.
    float  dist;       // distance from hit to the sampled point (shadow
                       //   ray t_max). 1e30 sentinel means "no hit"
                       //   already; not used here.
    uint   light_idx;  // which light_prims entry the sample came from --
                       //   the ReSTIR reservoir stores this so the
                       //   temporal/spatial merge passes re-evaluate
                       //   p_hat against the light that actually
                       //   produced the survivor.
};

// 5 float4s per light. Indexed by raw float4 lane offsets to keep the
// shader independent of how Slang lays the StructuredBuffer out on
// Metal (declaration-order MSL slot 11, vk::binding 27). v4 carries
// the rotation-gizmo orientation quaternion (xyzw); spot rotates its
// `dir` by it before the cone test, quad rotates both `dir` (normal)
// and the assembled u-extent at sample time. Identity (0,0,0,1)
// short-circuits to a no-op via quatRotate.
static const uint kLightStrideFloat4 = 5u;

struct LightRecord {
    float3 pos;
    float  radius;
    float3 intensity;
    uint   type;
    float3 dir;        // spot axis / quad normal (pre-rotation)
    float  cos_outer;  // spot only
    float3 u_or_inner; // quad u-half-extent vec (pre-rotation); spot: x = cos_inner
    float  v_half;     // quad v-axis half-extent (length)
    float4 orient;     // orientation quaternion (xyzw); identity = (0,0,0,1)
};

LightRecord loadLight(uint idx) {
    uint base = idx * kLightStrideFloat4;
    float4 v0 = light_prims[base + 0u];
    float4 v1 = light_prims[base + 1u];
    float4 v2 = light_prims[base + 2u];
    float4 v3 = light_prims[base + 3u];
    float4 v4 = light_prims[base + 4u];
    LightRecord L;
    L.pos        = v0.xyz;
    L.radius     = v0.w;
    L.intensity  = v1.rgb;
    L.type       = uint(v1.w + 0.5);
    L.dir        = v2.xyz;
    L.cos_outer  = v2.w;
    L.u_or_inner = v3.xyz;
    L.v_half     = v3.w;
    L.orient     = v4;
    return L;
}

// Uniform direction on the unit sphere from two uniform variates --
// pulled out as a tiny helper so the sphere-light sampler reads
// naturally. xi.x in [0,1] -> cos(theta) in [-1,1], xi.y -> phi.
float3 uniformSphereDirection(float2 xi) {
    float z   = 1.0 - 2.0 * xi.x;
    float r   = sqrt(max(0.0, 1.0 - z * z));
    float phi = 6.28318530717958 * xi.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Sample one analytic light (#73). Uniform over the count for the
// selection step (1/N); per-light sampling is per-type:
//   POINT  : delta function. pdf returned as 1.0 in solid angle; the
//             estimator's cos/pdf reduces to the inverse-square law
//             baked into Le below.
//   SPOT   : same as POINT, with an angular falloff window between
//             cos_inner and cos_outer.
//   SPHERE : uniform point on the surface (4*pi*r^2 area); pdf is
//             (area-pdf) * (dist^2 / cos_light), Veach 2.6 standard.
//   QUAD   : uniform point on the (2u) x (2v) rectangle defined by
//             pos +/- u +/- v_axis * v_half; same area-to-solid-angle
//             reciprocal.
//
// --- Light tree (#129) traversal -------------------------------------------
// 4 float4s per light tree node. Mirrors LightTreeNode in
// src/renderer/LightTree.h. The MSL/SPIR-V lane indexing is
// declaration-order-stable on Slang's targets so we just compute the
// raw float4 lane offset and read.
//
// PT_LIGHT_TREE_ENABLED gate: the loader/importance/picker helpers
// below are compiled out when the CMake option PT_LIGHT_TREE=OFF.
// Cuts ~150 lines of dead-but-emitted megakernel code that hurts
// ICache + register pressure on Apple Silicon even when the runtime
// cvar `r_light_tree=0` -- see perf/megakernel-split-light-tree.
// The runtime cvar continues to work as a no-op toggle in PT_LIGHT_TREE
// builds; in the disabled build the engine forces light_tree_enabled=0
// in PtPush so the uniform-pick fallback below is always taken.
#if PT_LIGHT_TREE_ENABLED
static const uint kLightTreeNodeStrideFloat4 = 4u;

// randf() (PathTraceMath.slang) returns float(seed & 0x00FFFFFF) / 2^24,
// i.e. a uniform draw from a 2^24-step lattice on [0, 1). The tree's
// per-level split probability is quantised onto that same lattice so the
// pdf the estimator divides by is the probability the sampler can
// actually realise -- see pickLightFromTree.
static const float kPickLattice = 16777216.0;          // 2^24
static const float kPickStep    = 1.0 / 16777216.0;    // 2^-24 (exact)

struct LightTreeNodeR {
    float3 aabb_min;
    uint   left_first;   // internal: left child index; leaf: light id
    float3 aabb_max;
    uint   count;        // 0 = internal, 1 = leaf
    float3 cone_axis;
    float  cone_cos_half;
    float  intensity;
};

LightTreeNodeR loadLightTreeNode(uint idx) {
    uint base = idx * kLightTreeNodeStrideFloat4;
    float4 v0 = light_tree_nodes[base + 0u];
    float4 v1 = light_tree_nodes[base + 1u];
    float4 v2 = light_tree_nodes[base + 2u];
    float4 v3 = light_tree_nodes[base + 3u];
    LightTreeNodeR n;
    n.aabb_min       = v0.xyz;
    n.left_first     = asuint(v0.w);
    n.aabb_max       = v1.xyz;
    n.count          = asuint(v1.w);
    n.cone_axis      = v2.xyz;
    n.cone_cos_half  = v2.w;
    n.intensity      = v3.x;
    return n;
}

// Per-node importance estimate at a shade point.  Returns a NON-NEGATIVE
// scalar; siblings are compared via ratio so absolute scale doesn't
// matter.  Conty Estevez & Kulla 2018 section 5.2: importance =
// emitted_power * geometric_falloff * orientation_term * receiver_cosine.
//
//   emitted_power     = node.intensity (scalar; sum of leaf intensities).
//   geometric_falloff = 1 / max(d^2, r^2) where d is the distance from
//                       the shade point to the cluster centre and r is
//                       half the cluster AABB diagonal -- bounds the
//                       inverse-square term so we don't blow up when
//                       the shade point is inside the cluster.
//   orientation_term  = cos(max(0, theta - theta_o - theta_u)) -- how far
//                       the direction cluster->shade_point sits outside
//                       the cluster's emission cone, widened by the angle
//                       the cluster subtends.
//   receiver_cosine   = cos(max(0, theta_i - theta_u)) -- the Lambert
//                       term at the receiver, widened the same way.
//
// #250: BOTH angular terms are BOUNDS OVER THE WHOLE CLUSTER, not point
// evaluations at the cluster centre.  theta_u is the half-angle the
// cluster's bounding sphere subtends from the shade point
// (sin theta_u = min(1, r / d)); subtracting it is what makes the
// estimate conservative, and it is the step Conty Estevez & Kulla spell
// out that the original port dropped.  The previous code tested the
// receiver cosine and the emission cone on the centre direction alone,
// so a cluster STRADDLING the receiver's tangent plane -- half its
// emitters above it, half below -- scored zero even while holding the
// brightest light in the scene, and a whole subtree's energy vanished
// from the image.
//
// Because the terms are now bounds, a return value of EXACTLY 0 is a
// proof rather than a heuristic: every emitter in the cluster is either
// behind the receiver's tangent plane or aimed away from it, so
// sampleAnalyticLight would reject each of them on `n_dot_l > 0` (or on
// the spot cutoff / quad one-sidedness) anyway.  Skipping such a subtree
// loses no energy -- which is why the old `max(..., 1e-8)` floor is gone.
// An ABSOLUTE floor sitting 8+ orders of magnitude under a live sibling
// is exactly what let float32 round the conditional split to 1.0 in
// pickLightFromTree below; a relative floor is unnecessary once zero
// means zero.
float lightTreeImportance(LightTreeNodeR node, float3 hit_pt,
                           float3 n_face) {
    float3 centre = 0.5 * (node.aabb_min + node.aabb_max);
    float3 to_c   = centre - hit_pt;
    float  d2     = max(dot(to_c, to_c), 1e-12);
    float  d      = sqrt(d2);
    float3 dir_to = to_c / d;

    // Bounding-sphere radius around the AABB; max distance from centre
    // to a corner = half the diagonal length.
    float3 ext    = 0.5 * (node.aabb_max - node.aabb_min);
    float  rad2   = max(dot(ext, ext), 1e-12);

    // Inverse-square falloff, clamped at the cluster radius so the
    // shade point being inside the cluster doesn't make the estimate
    // singular.
    float falloff = 1.0 / max(d2, rad2);

    // theta_u: half-angle subtended by the cluster's bounding sphere.
    // d <= r (shade point inside the sphere) saturates to pi/2, which
    // makes both angular terms fully permissive -- correct, because the
    // cluster then surrounds the receiver.
    float sin_u = min(1.0, sqrt(rad2) / d);
    float cos_u = sqrt(saturate(1.0 - sin_u * sin_u));

    // Orientation term. The cluster emits toward the shade point along
    // (-dir_to) from its own perspective, so theta is the angle between
    // the cone axis and that direction. The cone is widened by theta_u
    // before the test: alpha = theta_o + theta_u, evaluated through the
    // angle-sum identities so no inverse trig is needed.
    //   cos alpha = cos theta_o cos theta_u - sin theta_o sin theta_u
    //   sin alpha = sin theta_o cos theta_u + cos theta_o sin theta_u
    // sin alpha < 0 means alpha > pi, i.e. the widened cone covers every
    // direction -- which is also how the fully spherical sentinel
    // (cos_half = -1 -> theta_o = pi) falls out with no special case.
    float cos_emit = -dot(node.cone_axis, dir_to);
    float cos_o    = node.cone_cos_half;
    float sin_o    = sqrt(saturate(1.0 - cos_o * cos_o));
    float cos_a    = cos_o * cos_u - sin_o * sin_u;
    float sin_a    = sin_o * cos_u + cos_o * sin_u;
    float orient;
    if (sin_a < 0.0 || cos_emit >= cos_a) {
        orient = 1.0;               // inside the widened cone
    } else {
        float sin_emit = sqrt(saturate(1.0 - cos_emit * cos_emit));
        orient = max(0.0, cos_emit * cos_a + sin_emit * sin_a);
    }

    // Receiver-side Lambert bound: cos(max(0, theta_i - theta_u)).
    float cos_i = dot(n_face, dir_to);
    float recv;
    if (cos_i >= cos_u) {
        recv = 1.0;                 // the cluster spans the surface normal
    } else {
        float sin_i = sqrt(saturate(1.0 - cos_i * cos_i));
        recv = max(0.0, cos_i * cos_u + sin_i * sin_u);
    }

    return node.intensity * falloff * orient * recv;
}

// Top-down 2-way tree traversal.  At each internal node, computes the
// importance of each child and draws a uniform variate to pick one.
// Returns the leaf light id AND accumulates the selection PDF along
// the path (prod_i p_i where p_i is the conditional pick probability
// at each level). The shading-side PDF is `pick_pdf * solid_angle_pdf`
// just like the naive uniform pick -- only the selection PMF changes.
//
// The invariant that matters for correctness (#250): every light whose
// n_dot_l is positive at `hit_pt` must leave this walk with a STRICTLY
// POSITIVE pick_pdf. `radiance += Le * brdf * n_dot_l / ls.pdf` is an
// importance-sampling estimator -- a light handed probability zero is
// not down-weighted, it is deleted from the image.
struct LightPick {
    uint  light_id;
    float pick_pdf;
};

LightPick pickLightFromTree(float3 hit_pt, float3 n_face, inout uint seed) {
    LightPick r;
    r.light_id = 0u;
    r.pick_pdf = 1.0;

    uint idx = 0u;   // root
    // Cap depth defensively: the largest tree we build at MVP scale
    // (1000 lights) is ~10 deep; 32 is the cap the linker can handle
    // and well past any reasonable host-build.
    [unroll(32)]
    for (uint d = 0u; d < 32u; ++d) {
        LightTreeNodeR n = loadLightTreeNode(idx);
        if (n.count > 0u) {
            // Leaf: light id is in left_first.
            r.light_id = n.left_first;
            return r;
        }
        uint left  = n.left_first;
        uint right = left + 1u;
        LightTreeNodeR cl = loadLightTreeNode(left);
        LightTreeNodeR cr = loadLightTreeNode(right);
        float il = lightTreeImportance(cl, hit_pt, n_face);
        float ir = lightTreeImportance(cr, hit_pt, n_face);
        float p_l;
        if (il > 0.0 && ir > 0.0) {
            // BOTH children hold emitters that can reach the shade
            // point, so NEITHER may be handed probability zero -- the
            // estimator divides by pick_pdf, and a zero-probability
            // subtree is energy deleted, not energy resampled (#250).
            // float32 does exactly that on its own: il / (il + ir)
            // rounds to 1.0 as soon as ir drops below 2^-24 of il, and
            // `1.0 - p_l` is then exactly 0.
            //
            // Quantise the split onto randf()'s own lattice instead of
            // clamping with an invented epsilon. randf() returns
            // k * 2^-24 for k uniform in [0, 2^24), so the probability
            // the sampler can ACTUALLY realise for `u < p` is
            // ceil(p * 2^24) * 2^-24. Rounding p onto that lattice and
            // dividing by the rounded value makes pick_pdf the exact
            // probability of the branch we took -- the un-rounded form
            // was already biased whenever p fell below one lattice step
            // -- while ceil plus the [1, 2^24-1] clamp guarantee every
            // live subtree keeps at least one step of it.
            // Both m * kPickStep and randf()'s k * kPickStep are exact
            // in float32 (2^-24 is a power of two and m < 2^24), so
            // `u < p_l` is exactly `k < m`, with no rounding slack.
            float m = clamp(ceil((il / (il + ir)) * kPickLattice),
                            1.0, kPickLattice - 1.0);
            p_l = m * kPickStep;
        } else if (il > 0.0) {
            p_l = 1.0;              // right subtree provably contributes nothing
        } else if (ir > 0.0) {
            p_l = 0.0;              // left subtree provably contributes nothing
        } else {
            // Neither side can reach the shade point. Any split is
            // correct (every sample gets rejected on n_dot_l / the
            // emitter's own cutoff); 50/50 keeps the walk well defined
            // and the divide safe.
            p_l = 0.5;
        }
        float u   = randf(seed);
        if (u < p_l) {
            idx = left;
            r.pick_pdf *= p_l;
        } else {
            idx = right;
            r.pick_pdf *= (1.0 - p_l);
        }
    }
    return r;
}
#endif  // PT_LIGHT_TREE_ENABLED
// --- end Light tree traversal ----------------------------------------------

// Returns an unset sample (pdf=0) when light_count == 0 or the chosen
// type isn't a valid hemisphere direction (back-of-quad, outside spot
// cone). The caller treats pdf<=0 as "skip this NEE sample".
//
// Picker logic:
//   light_tree_enabled = 1 AND light_tree_node_count > 0 ->
//     traverse the hierarchical tree (Conty Estevez & Kulla 2018)
//     for O(log N) importance-weighted selection.
//   else -> uniform single pick over the active light_count (the
//     naive #73 sampler -- still correct, just higher variance).
//
// Both paths produce a `pick_pdf` that the per-light sampling stage
// multiplies into the solid-angle pdf; the shading math (Le * brdf *
// cos / pdf) is identical regardless.
LightSample sampleAnalyticLight(float3 hit_pt, float3 n_face, inout uint seed) {
    LightSample s;
    s.dir  = float3(0, 1, 0);
    s.Le   = float3(0.0);
    s.pdf  = 0.0;
    s.dist = 1e30;
    s.light_idx = 0u;
    if (light_count == 0u) return s;

    uint  pick;
    float pick_pdf;
#if PT_LIGHT_TREE_ENABLED
    // Light-tree path compiled in. The runtime cvar r_light_tree drives
    // light_tree_enabled/light_tree_node_count from the host so flipping
    // it to 0 still falls through to the uniform-pick branch at runtime,
    // with the tree-traversal code present-but-cold in the megakernel.
    if (light_tree_enabled != 0u && light_tree_node_count > 0u) {
        LightPick lp = pickLightFromTree(hit_pt, n_face, seed);
        pick     = min(lp.light_id, light_count - 1u);
        pick_pdf = lp.pick_pdf;
    } else
#endif  // PT_LIGHT_TREE_ENABLED
    {
        // Uniform single-pick over the active count. Same code path as
        // #73 -- regression-protected by the r_light_tree=0 cvar so we
        // can A/B variance against the tree at any time. In PT_LIGHT_TREE=OFF
        // builds this branch is *the only path*: the engine never sets
        // light_tree_enabled and the tree traversal is excised from the
        // megakernel entirely.
        float u_pick = randf(seed);
        pick = min(uint(u_pick * float(light_count)), light_count - 1u);
        pick_pdf = 1.0 / float(light_count);
    }

    LightRecord L = loadLight(pick);
    s.light_idx = pick;

    if (L.type == LIGHT_POINT) {
        float3 to_light = L.pos - hit_pt;
        float  dist2    = max(dot(to_light, to_light), 1e-12);
        float  dist     = sqrt(dist2);
        s.dir  = to_light / dist;
        // Inverse-square falloff: at distance d, irradiance from a
        // point source of intensity I (W/sr) is I/d^2. The estimator
        // below evaluates (Le * brdf * cos), with pdf=1 (delta light),
        // so we encode the geometric falloff into Le.
        s.Le   = L.intensity / dist2;
        // Delta light: any non-zero pdf works since cos/pdf collapses;
        // pick 1.0 so the MIS balance heuristic against a non-delta
        // BRDF sample naturally pushes toward this strategy.
        s.pdf  = pick_pdf;
        s.dist = dist;
        return s;
    }
    if (L.type == LIGHT_SPOT) {
        float3 to_light = L.pos - hit_pt;
        float  dist2    = max(dot(to_light, to_light), 1e-12);
        float  dist     = sqrt(dist2);
        float3 to_l_dir = to_light / dist;
        // Spot angular falloff: 1 inside cos_inner, 0 outside cos_outer,
        // smooth linear ramp in between. Direction from surface toward
        // the LIGHT is -to_l_dir from the light's emission lobe (the
        // spot axis points from emitter into the world). Rotate the
        // stored axis through the orientation quaternion so the editor
        // rotate gizmo can tilt the cone without mutating L.dir.
        // Identity quat (0,0,0,1) leaves the axis unchanged.
        float3 axis   = normalize(quatRotate(L.orient, L.dir));
        float cos_dir = dot(-to_l_dir, axis);
        float cos_in  = L.u_or_inner.x;
        float cos_out = L.cos_outer;
        float falloff = 0.0;
        if (cos_dir >= cos_in) {
            falloff = 1.0;
        } else if (cos_dir > cos_out) {
            falloff = (cos_dir - cos_out) / max(cos_in - cos_out, 1e-6);
        }
        s.dir  = to_l_dir;
        s.Le   = L.intensity * (falloff / dist2);
        s.pdf  = pick_pdf;
        s.dist = dist;
        return s;
    }
    if (L.type == LIGHT_SPHERE) {
        // Uniform-area sample on the surface of a sphere. A more
        // efficient sampler (uniform over the visible hemisphere from
        // the shade point) is an obvious follow-up; uniform-area gives
        // correct unbiased results -- just higher variance when the
        // shade point is near the sphere.
        float2 xi = float2(randf(seed), randf(seed));
        float3 n_l = uniformSphereDirection(xi);
        float3 lp  = L.pos + n_l * L.radius;
        float3 to_light = lp - hit_pt;
        float  dist2    = max(dot(to_light, to_light), 1e-12);
        float  dist     = sqrt(dist2);
        float3 wi       = to_light / dist;
        // Light surface normal points outward; only the hemisphere
        // facing the receiver contributes.
        float cos_light = max(0.0, dot(n_l, -wi));
        if (cos_light <= 0.0) return s;
        float area = 4.0 * 3.14159265358979 * L.radius * L.radius;
        // Solid-angle pdf = (1/area) * dist^2 / cos_light.
        // Selection pmf folds in via pick_pdf.
        float pdf_sa = (dist2) / max(area * cos_light, 1e-12);
        s.dir  = wi;
        // L.intensity carries area-emitter radiance (W/m^2/sr). The
        // outgoing radiance toward the receiver is just L.intensity
        // (no per-point modulation in our Lambertian-emitter model).
        s.Le   = L.intensity;
        s.pdf  = pick_pdf * pdf_sa;
        s.dist = dist;
        return s;
    }
    if (L.type == LIGHT_QUAD) {
        // Uniform-area on a (2u) x (2v) rectangle. The host packs
        // u_extent as the FULL half-extent vector (length = u_half);
        // v-axis is normalize(cross(normal, u)), and v_half is the
        // half-extent length. Both the normal and the u-extent vector
        // are rotated through the orientation quaternion so the
        // editor rotate gizmo can tilt the quad in-place. Identity
        // (0,0,0,1) leaves the stored frame untouched.
        float2 xi = float2(randf(seed) * 2.0 - 1.0,
                           randf(seed) * 2.0 - 1.0);   // [-1, 1]
        float3 u_vec = quatRotate(L.orient, L.u_or_inner);
        float  u_len = length(u_vec);
        if (u_len < 1e-6) return s;
        float3 n_quad = normalize(quatRotate(L.orient, L.dir));
        float3 v_axis = normalize(cross(n_quad, u_vec));
        float3 lp = L.pos + u_vec * xi.x + v_axis * (L.v_half * xi.y);
        float3 to_light = lp - hit_pt;
        float  dist2    = max(dot(to_light, to_light), 1e-12);
        float  dist     = sqrt(dist2);
        float3 wi       = to_light / dist;
        // One-sided emitter (front face is L.dir).
        float cos_light = dot(n_quad, -wi);
        if (cos_light <= 0.0) return s;
        float area = 4.0 * u_len * L.v_half;
        float pdf_sa = (dist2) / max(area * cos_light, 1e-12);
        s.dir  = wi;
        s.Le   = L.intensity;
        s.pdf  = pick_pdf * pdf_sa;
        s.dist = dist;
        return s;
    }
    return s;
}
// ---------- end Analytic light primitives NEE --------------------------

// RGB transmittance along a shadow ray. 1.0 = unoccluded, 0.0 = blocked
// by an opaque hit. Dielectric hits don't block: the ray refracts
// through (in -> out), continuing toward the light, multiplied by the
// dielectric's tint (cheap stand-in for Beer's-law absorption). This
// is what turns "shadow under a glass diamond" into actual caustic
// patterns -- the focal lensing emerges from the geometry the
// refracted shadow ray traces, not from any fancy sampling.
//
// --- Fluid Phase 1 (#136) -- smoke emitters --------------------------------
// Sums additive Gaussian-falloff density contributions from every
// active smoke emitter at the given world-space sample point. Returns
// 0 when smoke is disabled or no emitters are bound; otherwise
// extinction units (per metre, same as the cloud field's sigma_t).
//
// Phase 1 baseline: each emitter is a SINGLE Gaussian whose centre
// drifts as base + velocity * t. Reads as a translating blob.
//
// --- Fluid Phase 2 (#180) -- advected emission -----------------------------
// Each emitter is now a CHAIN of `puff_count` Gaussians spaced evenly
// across [0, r_smoke_lifetime] in age. Puff k (k in [0..P-1]) has:
//   * age_k       = (k / max(P-1, 1)) * lifetime           (seconds)
//   * pos_k       = base + velocity * advection_strength * age_k
//   * radius_k    = radius * (1 + expansion_rate * age_k)
//   * density_k   = peak * (1 - age_k / lifetime)         (linear fade)
// The head puff (k=0, age=0) sits at the emitter mouth at full density;
// the tail puff (k=P-1, age=lifetime) has zero density and the farthest
// position. Result reads as a tapering, dissipating column rather than
// a rigid translating blob.
//
// Fallback: when lifetime <= 0 or puff_count <= 1 the inner loop runs
// exactly once with age=0, recovering the Phase 1 single-blob shape
// bit-exact. This is the default-off escape hatch and what an existing
// scene file with no Phase 2 cvars set gets out-of-the-box.
//
// Falloff: density_k * exp(-(r/radius_k)^falloff) -- with falloff=2
// this is a true Gaussian; falloff=1 collapses to exp(-r/radius), a
// steeper near-source profile; higher exponents harden the core /
// soften the wings (useful for soot vs steam).
//
// Capped at 16 emitters x 16 puffs via the kMaxSmokeEmitters / kSmokePuffMax
// constants; the host further clamps via r_smoke_max_emitters and
// r_smoke_puff_count. Inner puff loop is `[loop]` because the bound is
// push-driven (varies per dispatch); Slang's MSL backend produces tight
// code at N<=16.
//
// Wave-6 "looks like smoke" upgrades:
//   * kSmokePuffMax bumped from 8 -> 16 so a column reads as continuous
//     rather than as a string of discrete blobs.
//   * Per-puff density modulated by a 3-octave fbm (cloud_fbm) so the
//     Gaussian core gets natural wispy turbulence detail instead of a
//     smooth blob look.
//   * Age-fade curve takes an exponent (smoke_params3.z) so the head
//     stays dense and the tail wisps away faster than the legacy linear
//     fade; exp=1 recovers Phase 2's bit-exact (1 - age/lifetime) shape.
static const int kSmokePuffMax = 16;

// Smoke-specific 3-octave fbm sampling using the cloud_fbm helper imported
// from PathTraceCloud. Sampled in WORLD space (not puff-local) so adjacent
// overlapping puffs see DECORRELATED noise contributions -- this is what
// lets the chain of puffs read as a single continuous turbulent column
// rather than a stack of identically-textured blobs.
//
// Returns a noise value in roughly [0..1]. Sample positions are scaled
// by `freq` (cycles per metre at the base radius) which the host pushes
// via smoke_params3.x; freq=4 reads as fine high-frequency wisps.
float smoke_turbulence_noise(float3 world_pos, float age, float freq) {
    // Domain-warp by age in seconds: the warp offset is small relative
    // to the noise period so older puffs see a slightly shifted slice
    // of the underlying field. Combined with the world-space input this
    // gives each puff its own portion of noise, so overlap blends
    // multi-frequency rather than just multiplying the same pattern.
    float3 warp = float3(age * 0.7, age * 1.3, age * 0.5);
    float3 q = world_pos * freq + warp;
    // 3 octaves: enough for fine wisps without inflating the per-sample
    // cost dramatically. Octave count fixed at compile time (not a
    // cvar) -- the visual difference between 3 and 4 is small but the
    // perf delta is ~25%; keep it tight.
    return cloud_fbm(q, 3);
}

// 3D curl-style noise displacement for domain-warping the smoke sample
// position. Returns a vector roughly in [-1..1]^3 that we displace the
// sample position by before density evaluation. Three decorrelated
// noise taps at large offsets -- similar idea to cloud_curl_noise in
// PathTraceCloud.slang but cheap (one fbm per component, no central
// differences) since visual divergence-freeness doesn't matter for the
// smoke silhouette breakup. The point is to PERTURB the Gaussian's
// centre as seen from a given world point so the COMBINED chain of
// puffs no longer reads as a smooth oval body.
float3 smoke_warp_offset(float3 world_pos, float age, float freq) {
    float3 q = world_pos * freq + float3(age * 0.5, age * 0.8, age * 1.1);
    float nx = cloud_fbm(q + float3(  0.0,   0.0,   0.0), 2);
    float ny = cloud_fbm(q + float3(173.7, 211.3,  31.7), 2);
    float nz = cloud_fbm(q + float3(311.1,  47.3, 593.9), 2);
    // Each octave-2 fbm returns [0..1]; centre to [-1..1] then return.
    return float3(2.0 * nx - 1.0, 2.0 * ny - 1.0, 2.0 * nz - 1.0);
}

float smoke_density_add(float3 pos) {
    float smoke_on = smoke_params.y;
    if (smoke_on <= 0.0) return 0.0;
    int n = int(smoke_params.x);
    if (n <= 0) return 0.0;
    float t = smoke_params.z;
    // Hard cap so the loop bound is a compile-time-known maximum --
    // matches kMaxSmokeEmitters on the host. Without this the MSL
    // backend has to assume an unbounded loop and emits a less-tight
    // bounds check per iteration.
    if (n > 16) n = 16;
    // Phase 2 (#180) puff-chain parameters. Clamped on the host but
    // re-bounded here so a runtime cvar push that arrived early can't
    // index out of the [1..kSmokePuffMax] range. lifetime <= 0
    // collapses the chain to a single head puff (Phase 1 fallback).
    float lifetime   = smoke_params2.x;
    float adv_strength = smoke_params2.y;
    int   puff_count = int(smoke_params2.z + 0.5);
    if (puff_count < 1) puff_count = 1;
    if (puff_count > kSmokePuffMax) puff_count = kSmokePuffMax;
    float expansion = max(smoke_params2.w, 0.0);
    bool  chain     = (lifetime > 0.0) && (puff_count > 1);
    // Wave-6 turbulence + decay-shape knobs (smoke_params3).
    float turb_freq  = max(smoke_params3.x, 0.0);
    float turb_amp   = clamp(smoke_params3.y, 0.0, 1.0);
    float decay_exp  = max(smoke_params3.z, 1.0);
    float dens = 0.0;
    [loop] for (int i = 0; i < n; ++i) {
        float4 v0 = smoke_emitters[i * 3 + 0];
        float4 v1 = smoke_emitters[i * 3 + 1];
        float4 v2 = smoke_emitters[i * 3 + 2];
        float3 base = v0.xyz;
        float  radius = max(v0.w, 1e-3);   // guard against div-by-zero
        float3 vel = v1.xyz;
        float  peak = v1.w;
        float  falloff = max(v2.w, 0.25);  // shaping; clamp to keep stable
        // Cap loop bound by puff_count when chaining; collapse to 1
        // for the Phase 1 fallback path. Hoisting these into local
        // scope lets the optimiser fold the early-out for the common
        // chain=false case.
        int    P    = chain ? puff_count : 1;
        float  span = chain ? lifetime    : 1.0;
        // Sweep a global phase across the lifetime so the chain itself
        // visibly cycles forward over time -- without this, every
        // frame samples the SAME age set (just at different drifted
        // centres if the emitter has velocity), which is fine for a
        // stretched static column but kills the "rising puffs"
        // animation the user expects when watching a capture sequence.
        // The phase wraps in [0, span/P) so neighbouring puffs leap-frog
        // into each other's slot once per cycle, producing a continuous
        // upward / outward flow.
        float pstep = (P > 1) ? (span / float(P - 1)) : 0.0;
        float t_offset = (pstep > 0.0) ? (t - pstep * floor(t / pstep)) : 0.0;
        // Cheap early-out: bound the entire chain by the head + tail
        // bounding sphere. Tail centre = base + vel * adv * lifetime;
        // tail radius = radius * (1 + expansion * lifetime). Test the
        // sample against an expanded sphere encompassing both ends --
        // misses skip the inner loop entirely, which is the dominant
        // empty-space case.
        float3 tail_c   = base + vel * (adv_strength * span);
        float  tail_r   = radius * (1.0 + expansion * span);
        float3 mid_c    = 0.5 * (base + tail_c);
        float  half_ext = 0.5 * length(tail_c - base) + max(radius, tail_r) * 4.0;
        float3 md       = pos - mid_c;
        if (dot(md, md) > half_ext * half_ext) continue;
        [loop] for (int k = 0; k < P; ++k) {
            float a_norm = (P > 1) ? (float(k) / float(P - 1)) : 0.0;
            float age    = a_norm * span + t_offset;
            if (age > span) age -= span;   // wrap-around cycle
            // Age-fade curve. peak * pow(1 - age/lifetime, decay_exp).
            // decay_exp = 1 recovers Phase 2's linear fade bit-exact;
            // decay_exp = 2 (default) keeps heads denser and tails wisp
            // away faster -- the characteristic "fresh puff, vanishing
            // tail" shape of real smoke columns.
            // No chain => density is just peak (Phase 1 bit-exact).
            float a_t  = (span > 0.0) ? clamp(age / span, 0.0, 1.0) : 0.0;
            float dfade = chain ? pow(1.0 - a_t, decay_exp) : 1.0;
            float3 cur  = base + vel * (adv_strength * age);
            float  rad  = radius * (1.0 + expansion * age);
            // Wave-6 domain warp: displace the sample position before
            // the puff-Gaussian lookup so each puff's silhouette is
            // BENT by the noise field rather than just modulated. This
            // is the missing piece that makes the COMBINED chain of
            // puffs look like turbulent wisps instead of a smooth oval
            // (per-puff multiplicative noise alone gets averaged away
            // by 16 overlapping Gaussians; warping the LOOKUP position
            // affects every puff that sees this world point, so the
            // perturbation accumulates rather than cancelling).
            // Warp magnitude grows with age so head puffs (the freshest
            // source) stay relatively crisp and tail puffs get strongly
            // perturbed -- real smoke columns are sharp at the source
            // and increasingly diffuse upward.
            float3 pos_warped = pos;
            if (turb_amp > 0.0 && turb_freq > 0.0) {
                float warp_mag = rad * turb_amp * (0.3 + 0.7 * a_t);
                float3 ofs = smoke_warp_offset(pos, age, turb_freq * 0.5);
                pos_warped = pos + ofs * warp_mag;
            }
            float3 d    = pos_warped - cur;
            float  r2   = dot(d, d);
            float cutoff = 16.0 * rad * rad;
            if (r2 > cutoff) continue;
            float r = sqrt(r2);
            float u = r / rad;
            // Generalised "Gaussian": exp(-u^falloff). For falloff=2
            // this is the canonical Gaussian; falloff=1 = exp(-u)
            // (sharper near-source); higher values harden the core.
            float w = exp(-pow(u, falloff));
            // Wave-6 turbulence modulation. Sample noise in WORLD space
            // (not puff-local) so overlapping puffs see decorrelated
            // wisps -- this is what produces visible internal structure
            // when the puff chain overlaps densely. amp=0 collapses to
            // Phase 2 bit-exact; amp=1 lets noise fully ERODE the
            // Gaussian wings into wisps. freq is in cycles/m for
            // intuitive scene tuning. The per-puff age adds a small
            // domain warp so the column appears to flow rather than
            // being pinned to world coords.
            //
            // EDGE erosion: weight the modulation by (1-w)^2 so the
            // dense core (w~1) is barely modulated -- preserves a
            // recognisable column body -- but the soft Gaussian outer
            // wings (small w, where w*turb dominates the silhouette)
            // get the FULL amp range of noise subtraction. Below the
            // erosion threshold the density goes negative and gets
            // clamped to 0, producing the wispy fingered silhouette
            // of real smoke vs. a smooth Gaussian oval. Same
            // (1-density)^2 wing-mask idea as cloud_density_at's
            // hi-frequency edge erosion (PathTraceCloud.slang).
            float turb_contrib = peak * dfade * w;
            if (turb_amp > 0.0 && turb_freq > 0.0) {
                float n_val = smoke_turbulence_noise(pos, age, turb_freq);
                float edge  = (1.0 - w) * (1.0 - w);
                // n_val [0..1] -> bias [-1..+1]; multiplied by amp and
                // edge to keep dense cores stable. Cores keep ~peak*w,
                // wings get sculpted into wisps.
                float bias  = (2.0 * n_val - 1.0) * turb_amp * edge;
                // Center sample modulation (small w, large edge) can
                // push density well below 0, which max() clamps. Body
                // samples (large w, edge~0) are nearly unmodulated.
                turb_contrib = max(0.0, turb_contrib * (1.0 + bias) - turb_amp * edge * 0.3);
            }
            dens += turb_contrib;
        }
    }
    return dens;
}

// Tinted variant used at the primary cloud-march sample point so that
// per-emitter `tint` (v2.xyz on the smoke SSBO) actually modulates the
// in-scattered radiance for shaded smoke. Pure shadow paths
// (cloud_optical_depth, cloud_optical_depth_stratified) stay on the
// scalar density-only version above -- the tint is a multiplier on
// the radiance the smoke scatters TOWARD the camera, not on the
// transmittance the smoke imposes on the sun NEE leg.
//
// Tint is a density-weighted average of the per-emitter tints sampled
// at `pos`. When a single plume dominates the sample the returned
// `tint` is that emitter's tint; when two plumes overlap the result
// is the in-between blend, matching how a real heterogeneous plume
// would emit a mixed-colour glow.
//
// out_tint is left at float3(1) when there's no smoke contribution at
// this sample, so a caller that lerps host_lighting * out_tint by the
// smoke fraction stays a no-op in clear air.
//
// Phase 2 (#180): inner loop expands each emitter into a chain of
// puffs; see smoke_density_add() above for the chain semantics. Tint
// weighting accumulates density (peak * fade * gaussian) so the
// brightest-density puff dominates the per-sample mix -- matches the
// physical "the densest, freshest part of the plume colours the
// transmitted radiance the most" intuition.
float smoke_density_add_tinted(float3 pos, out float3 out_tint) {
    out_tint = float3(1.0, 1.0, 1.0);
    float smoke_on = smoke_params.y;
    if (smoke_on <= 0.0) return 0.0;
    int n = int(smoke_params.x);
    if (n <= 0) return 0.0;
    float t = smoke_params.z;
    if (n > 16) n = 16;
    float lifetime     = smoke_params2.x;
    float adv_strength = smoke_params2.y;
    int   puff_count   = int(smoke_params2.z + 0.5);
    if (puff_count < 1) puff_count = 1;
    if (puff_count > kSmokePuffMax) puff_count = kSmokePuffMax;
    float expansion = max(smoke_params2.w, 0.0);
    bool  chain     = (lifetime > 0.0) && (puff_count > 1);
    // Wave-6 turbulence + decay-shape + age-tint knobs.
    float turb_freq    = max(smoke_params3.x, 0.0);
    float turb_amp     = clamp(smoke_params3.y, 0.0, 1.0);
    float decay_exp    = max(smoke_params3.z, 1.0);
    float age_tint_str = clamp(smoke_params3.w, 0.0, 1.0);
    float  dens     = 0.0;
    float3 tint_acc = float3(0.0, 0.0, 0.0);
    [loop] for (int i = 0; i < n; ++i) {
        float4 v0 = smoke_emitters[i * 3 + 0];
        float4 v1 = smoke_emitters[i * 3 + 1];
        float4 v2 = smoke_emitters[i * 3 + 2];
        float3 base = v0.xyz;
        float  radius = max(v0.w, 1e-3);
        float3 vel = v1.xyz;
        float  peak = v1.w;
        float3 tint = v2.xyz;
        float  falloff = max(v2.w, 0.25);
        int    P    = chain ? puff_count : 1;
        float  span = chain ? lifetime    : 1.0;
        float pstep = (P > 1) ? (span / float(P - 1)) : 0.0;
        float t_offset = (pstep > 0.0) ? (t - pstep * floor(t / pstep)) : 0.0;
        float3 tail_c   = base + vel * (adv_strength * span);
        float  tail_r   = radius * (1.0 + expansion * span);
        float3 mid_c    = 0.5 * (base + tail_c);
        float  half_ext = 0.5 * length(tail_c - base) + max(radius, tail_r) * 4.0;
        float3 md       = pos - mid_c;
        if (dot(md, md) > half_ext * half_ext) continue;
        [loop] for (int k = 0; k < P; ++k) {
            float a_norm = (P > 1) ? (float(k) / float(P - 1)) : 0.0;
            float age    = a_norm * span + t_offset;
            if (age > span) age -= span;
            float a_t   = (span > 0.0) ? clamp(age / span, 0.0, 1.0) : 0.0;
            // Wave-6: pow(1-age, decay_exp) tail-wisp shape (see
            // smoke_density_add for rationale; decay_exp=1 is bit-exact
            // legacy linear fade).
            float dfade = chain ? pow(1.0 - a_t, decay_exp) : 1.0;
            float3 cur  = base + vel * (adv_strength * age);
            float  rad  = radius * (1.0 + expansion * age);
            // Wave-6 domain warp: see smoke_density_add for rationale.
            // Keep both functions identical so the tinted in-scatter
            // path and the shadow path agree on silhouette geometry.
            float3 pos_warped = pos;
            if (turb_amp > 0.0 && turb_freq > 0.0) {
                float warp_mag = rad * turb_amp * (0.3 + 0.7 * a_t);
                float3 ofs = smoke_warp_offset(pos, age, turb_freq * 0.5);
                pos_warped = pos + ofs * warp_mag;
            }
            float3 d    = pos_warped - cur;
            float  r2   = dot(d, d);
            float cutoff = 16.0 * rad * rad;
            if (r2 > cutoff) continue;
            float r = sqrt(r2);
            float u = r / rad;
            float w = exp(-pow(u, falloff));
            // Wave-6 turbulence modulation (see smoke_density_add for
            // shape rationale; amp=0 collapses to Phase 2 bit-exact).
            // Edge-erosion form: wings get sculpted into wisps, core
            // stays stable.
            float di = peak * dfade * w;
            if (turb_amp > 0.0 && turb_freq > 0.0) {
                float n_val = smoke_turbulence_noise(pos, age, turb_freq);
                float edge  = (1.0 - w) * (1.0 - w);
                float bias  = (2.0 * n_val - 1.0) * turb_amp * edge;
                di = max(0.0, di * (1.0 + bias) - turb_amp * edge * 0.3);
            }
            // Wave-6 age tint gradient. Lerps the per-emitter base tint
            // toward a slightly cooler grey-blue as the puff ages -- the
            // characteristic real-world look of older smoke that has
            // drifted up and cooled. age_tint_str=0 collapses to
            // legacy behaviour (per-puff tint == per-emitter tint).
            // Fresh puffs (a_t~0) get a faintly WARM tint; old puffs
            // (a_t~1) get a faintly COOL tint. Centred so the
            // density-weighted average roughly matches the legacy mean
            // and exposure stays comparable.
            float3 warm = float3(1.06, 1.00, 0.93);   // subtle yellow
            float3 cool = float3(0.93, 0.96, 1.05);   // subtle blue-grey
            float3 age_mod = lerp(warm, cool, a_t);
            float3 puff_tint = lerp(tint, tint * age_mod, age_tint_str);
            dens     += di;
            tint_acc += di * puff_tint;
        }
    }
    if (dens > 0.0) out_tint = tint_acc / dens;
    return dens;
}
// --- end Fluid Phase 1 / Phase 2 -------------------------------------------

// --- Fluid Phase 3 (#22) -- SPH particle density splat ---------------------
// Sums Gaussian splats from the SPH particle list at world-space sample
// `pos`. Each particle is a small Gaussian with falloff exp(-(r/rad)^2);
// the integration over the ray-march body still happens in the
// cloud-march loop above, this function just contributes a scalar
// density at the sample point.
//
// The CPU SPH solver (pt::sph::SmokeSPH) integrates the particle pool
// against pressure / viscosity / gravity / buoyancy / wind / shockwaves
// each engine Tick; the GPU only sees the pos/radius/density triple
// per live particle and treats each as a sub-emitter. There's no per-
// particle ageing in this function -- the host already mapped age to
// the splat density before upload (older particles fade) so the shader
// can stay branch-light.
//
// Cost: O(N) per sample. For N = 1024 default cap, an empty-space
// cull (early-out on AABB-against-pos distance) keeps the cost flat
// in the common case where most particles are far from any given
// march sample. The early-out lives inside the loop body as a
// cutoff-vs-r2 test (cheap when the smoke column is compact).
//
// smoke_sph_params.y = mode gate: 0 = OFF (function returns 0; the
// procedural smoke_density_add still drives the field), 1 = SPH ONLY
// (procedural is bypassed by the host's smoke_params.y override), or
// 2 = BOTH (additive). The gate happens at the call site, not here --
// this function unconditionally accumulates whatever's in the buffer.

// --- Wave 8 polish (#28) ---------------------------------------------------
// SPH plume read fix: the Phase 3 splat used a wide soft Gaussian
// (exp(-(r/rad)^2)) which, summed over ~800-2048 heavily-overlapping
// particles, smeared the column into a uniform haze with no coherent
// vertical structure (the PR flagged "renders as a hazy cloud rather than
// a sharp columnar plume"). Tighten the kernel so each particle's density
// concentrates in its core: a steeper Gaussian exp(-kSphSplatSharp * u2)
// shrinks the effective FWHM by sqrt(kSphSplatSharp), so neighbouring
// splats overlap in a dense core + fall off fast at the edges rather than
// blending into fog. kSphSplatSharp = 3.0 reads as a coherent plume at the
// default particle radius without going grainy. 3-sigma cutoff is kept
// (the steeper kernel is ~0 well before it). Returns the per-particle
// splat weight for squared normalised radius u2 = r^2 / rad^2.
//
// kSphSplatGain compensates the optical-depth loss from sharpening: a
// steeper Gaussian has a smaller integral, so without a gain the column
// would go MORE transparent (the opposite of "coherent plume"). The gain
// boosts the peak so the plume core becomes optically thick enough to
// OCCLUDE the bright sky behind it -- the Beer-Lambert occlusion is what
// reads as a solid smoke column; a thin in-scatter-only column just
// washes into a bright sky. 4x is tuned so the plume core goes optically
// thick (Beer-Lambert transmittance -> ~0 through the column), letting
// the smoke's own grey in-scatter colour replace the bright sky behind it
// rather than glowing translucently over it.
static const float kSphSplatSharp = 3.0;
static const float kSphSplatGain  = 4.0;
// --- Wave 9 sph-3b -------------------------------------------------------
// Phase 3b: the single sharpened Gaussian (Wave 8) still read as a thin
// radial haze (no opaque body, no recognizable column). The fix is a
// 2-LOBE splat -- a dense narrow CORE plus a soft wide HALO -- so the
// plume core goes optically thick (recognizable opaque smoke body) while
// the edges stay wispy and the halo fills the gaps BETWEEN particles so
// the column reads as one continuous body rather than a cloud of discrete
// blobs.
//
//   * Core lobe: high amplitude (kSphCoreGain), steep falloff
//     (kSphCoreSharp) -> a tight, very dense kernel. Summed over the
//     overlapping near-emitter particles this drives the eye-ray
//     transmittance through the column core to ~0 (Beer-Lambert
//     occlusion), which is what makes the body read as solid smoke
//     instead of a translucent glow over the bright sky.
//   * Halo lobe: low amplitude (kSphHaloGain), gentle falloff
//     (kSphHaloSharp) -> a broad soft skirt out to the 3-sigma cutoff.
//     It bridges inter-particle gaps so the surface stays continuous and
//     gives the wispy fade at the column edge.
//
// The two-lobe sum has a markedly higher peak-to-skirt ratio than a
// single Gaussian of the same support, which is exactly the "dense core +
// wispy edge" profile that reads as a plume. Tuned at the smoke_phase3
// default particle radius; the 3-sigma cutoff (9*rad^2) is unchanged so
// the per-sample particle culling cost is identical to Wave 8.
static const float kSphCoreSharp = 7.0;   // steep -> tight dense core
static const float kSphCoreGain  = 7.0;   // tall   -> optically-thick body
static const float kSphHaloSharp = 1.1;   // gentle -> broad soft skirt
static const float kSphHaloGain  = 0.9;   // low    -> wispy continuous edge
float sph_splat_weight(float u2) {
    // exp(-7*u2) core for the opaque body + exp(-1.1*u2) halo for the
    // gap-filling wispy skirt. (Legacy Wave-8 kSphSplatSharp/Gain kept
    // referenced below so a single-lobe fallback stays trivially
    // re-derivable.)
    return kSphCoreGain * exp(-kSphCoreSharp * u2)
         + kSphHaloGain * exp(-kSphHaloSharp * u2);
}
// Reference the Wave-8 constants so they don't warn as unused while the
// 2-lobe kernel is the active path (kept for documentation / fallback).
static const float kSphSplatLegacyRef = kSphSplatGain * 0.0 + kSphSplatSharp * 0.0;
// --- end Wave 9 sph-3b ---------------------------------------------------
// --- end Wave 8 polish (#28) -----------------------------------------------

float smoke_sph_density_add(float3 pos) {
    int n = int(smoke_sph_params.x);
    if (n <= 0) return 0.0;
    // Hard cap mirrors the host upload cap (r_smoke_sph_max_particles).
    // The shader doesn't trust the runtime push value beyond a sanity
    // bound; misconfigured host could push n > buffer capacity and we
    // don't want a SSBO out-of-bounds read.
    int cap = int(smoke_sph_params.w);
    if (cap > 0 && n > cap) n = cap;
    if (n > 4096) n = 4096;
    float floor_d = smoke_sph_params.z;
    float dens = 0.0;
    [loop] for (int i = 0; i < n; ++i) {
        float4 v0 = sph_particles[i * 2 + 0];
        float4 v1 = sph_particles[i * 2 + 1];
        float3 pp = v0.xyz;
        float  rad = max(v0.w, 1e-3);
        float  peak = max(v1.x - floor_d, 0.0);
        if (peak <= 0.0) continue;
        float3 d = pos - pp;
        float r2 = dot(d, d);
        float cutoff = 9.0 * rad * rad;   // 3-sigma; outside ~zero contribution
        if (r2 > cutoff) continue;
        // Wave 8 (#28): sharpened Gaussian splat (sph_splat_weight) so the
        // column reads as a coherent plume instead of overlapping soft haze.
        float u2 = r2 / (rad * rad);
        float w  = sph_splat_weight(u2);
        dens += peak * w;
    }
    return dens;
}

// Tinted variant for the in-scatter path. Reads the same particle
// buffer and returns the density-weighted average per-particle tint via
// out_tint. Wave 8 (#28): the tint is now an age gradient (bright white
// for fresh near-emitter particles -> dim cool grey for old diluted
// ones, driven by the host-baked v1.y normalised-age scalar) so the
// column gets vertical structure -- a dense white base fading to a wispy
// grey crown. Keep the API parallel to smoke_density_add_tinted so the
// call site in PathTrace can blindly add the two contributions.
float smoke_sph_density_add_tinted(float3 pos, out float3 out_tint) {
    out_tint = float3(1.0, 1.0, 1.0);
    int n = int(smoke_sph_params.x);
    if (n <= 0) return 0.0;
    int cap = int(smoke_sph_params.w);
    if (cap > 0 && n > cap) n = cap;
    if (n > 4096) n = 4096;
    float floor_d = smoke_sph_params.z;
    float dens = 0.0;
    float3 tint_acc = float3(0.0, 0.0, 0.0);
    [loop] for (int i = 0; i < n; ++i) {
        float4 v0 = sph_particles[i * 2 + 0];
        float4 v1 = sph_particles[i * 2 + 1];
        float3 pp = v0.xyz;
        float  rad = max(v0.w, 1e-3);
        float  peak = max(v1.x - floor_d, 0.0);
        if (peak <= 0.0) continue;
        float3 d = pos - pp;
        float r2 = dot(d, d);
        float cutoff = 9.0 * rad * rad;
        if (r2 > cutoff) continue;
        // Wave 8 (#28): sharpened splat (matches smoke_sph_density_add).
        float u2 = r2 / (rad * rad);
        float w  = sph_splat_weight(u2);
        float di = peak * w;
        // Wave 8 (#28): age-driven tint COHESION for vertical plume
        // structure. Younger particles (near the emitter, a_norm -> 0)
        // read bright/white -- freshly-lofted hot smoke is optically dense
        // and reflective; older particles (a_norm -> 1) read as a dimmer,
        // desaturated grey as the puff dilutes and cools. This gives the
        // column a white dense base fading to a wispy grey crown rather
        // than a uniform tint top-to-bottom. The host bakes the density
        // age-fade into v1.x (PackForGpu) so this is purely the colour
        // gradient; the two together produce the columnar read.
        float a_norm  = saturate(v1.y);
        // Age-driven tint cohesion (Wave 8 #28). Fresh near-emitter
        // particles read bright + faintly warm (hot, optically dense
        // smoke catching the sun); aged/diluted particles fade to a
        // dimmer cool grey wisp. The brightness gradient (paired with the
        // density age-fade baked into v1.x by PackForGpu) gives the plume
        // a bright coherent base fading to a grey crown -- the vertical
        // structure that turns a flat blob into a recognisable plume.
        float3 fresh  = float3(1.15, 1.10, 1.02);   // bright warm-white base
        float3 old    = float3(0.55, 0.57, 0.62);   // dim cool grey crown
        float3 ptint  = lerp(fresh, old, a_norm);
        dens     += di;
        tint_acc += di * ptint;
    }
    if (dens > 0.0) out_tint = tint_acc / dens;
    return dens;
}
// --- end Fluid Phase 3 -----------------------------------------------------

// Ray versus the cloud layer, in whichever frame the scene is in
// (planetary P3, #257).
//
// PLANAR frame (planet_center_radius.w <= 0, every scene authored
// before this phase): `base` / `top` are world-Y planes and this is the
// slab intersection that has always been here, expression for
// expression -- so the planar path is bit-for-bit unchanged.
//
// SPHERICAL frame: `base` / `top` are ALTITUDES above the surface and
// the layer is a shell of radii R+base .. R+top about the planet centre.
// That is the fix for #51's founding symptom. A slab is a pair of
// infinite planes, so a perfectly horizontal ray from a ground camera
// needs infinite horizontal distance to reach a layer above it: clouds
// stopped at a hard line with a strip of bare sky underneath, and could
// never touch the horizon. On a shell the same ray leaves through the
// top of the layer at a finite range, because the ground curves away
// beneath it, and the layer terminates exactly at the horizon.
//
// Returns false when the ray never enters the layer.
bool cloudLayerInterval(float3 ro, float3 rd, float base, float top,
                        float t_max, out float t_in, out float t_out) {
    if (planet_center_radius.w > 0.0) {
        float R = planet_center_radius.w;
        return ptRayShell(ro, rd, planet_center_radius.xyz,
                          R + base, R + top, t_max, t_in, t_out);
    }
    float ry = rd.y;
    if (abs(ry) > 1e-6) {
        float t_b = (base - ro.y) / ry;
        float t_t = (top  - ro.y) / ry;
        t_in  = max(0.0, min(t_b, t_t));
        t_out = min(max(t_b, t_t), t_max);
    } else if (ro.y >= base && ro.y <= top) {
        t_in = 0.0; t_out = t_max;
    } else {
        t_in = 1.0; t_out = 0.0;
        return false;
    }
    return t_out > t_in;
}

// Optical depth through the cloud layer along [ro, ro + rd*t_max].
// Returns 0 when clouds are off or the ray misses the layer; otherwise
// integrates cloud_density_at across the ray-slab intersection. Used
// by transmittance() to give clouds genuine shadow-casting behaviour:
// any NEE shadow ray that grazes a dense cloud loses light.
float cloud_optical_depth(float3 ro, float3 rd, float t_max) {
    // Fluid Phase 1 (#136): also accumulate smoke density on the
    // shadow path so opaque smoke blocks sunlight through to the
    // ground, matching the primary-march visibility.
    // Smoke is "active" when either the procedural Phase 2 emitters
    // are bound (smoke_params.y gate + count) OR the Phase 3 SPH
    // particles are bound (smoke_sph_params.x > 0). The host's mode
    // selection in r_smoke_mode collapses one of the two to zero so
    // only one path contributes per frame; the OR keeps the gate
    // logic generic.
    bool smoke_active = (smoke_params.y > 0.0) && (smoke_params.x > 0.0);
    bool sph_active   = (smoke_sph_params.x > 0.0);
    smoke_active = smoke_active || sph_active;
    if (clouds_p1.w <= 0.0 && !smoke_active) return 0.0;
    float base_y = clouds_p1.x;
    float top_y  = clouds_p1.y;
    if (clouds_p1.w <= 0.0) {
        base_y = 0.0;
        top_y  = 2000.0;
    }
    float t_in, t_out;
    if (!cloudLayerInterval(ro, rd, base_y, top_y, t_max, t_in, t_out)) {
        return 0.0;
    }
    // Cap shadow march distance: a near-horizon ray crossing the cloud
    // layer at very shallow elevation can span hundreds of kilometres,
    // far more than the cloud field's coherence length. 20km is plenty
    // to capture the shadow of any reasonable cloud bank in front of
    // the sun, and limits the sample budget. The NEAR window is kept --
    // t_in is the entry nearest the ray origin.
    if (t_out - t_in > 20000.0) t_out = t_in + 20000.0;
    if (t_out <= t_in) return 0.0;
    // 4 coarse samples + 3 fbm octaves for the shadow ray. Beer-Lambert
    // exp(-OD) only needs a rough density estimate to look right.
    const int Ns = 4;
    float dt = (t_out - t_in) / float(Ns);
    float od = 0.0;
    [unroll] for (int i = 0; i < Ns; ++i) {
        float t = t_in + (float(i) + 0.5) * dt;
        float3 sp = ro + rd * t;
        float dens = cloud_density_at(sp, clouds_p1, clouds_p2, clouds_p3,
                                       clouds_p4, 3, planet_center_radius);
        // Fluid Phase 1 (#136): smoke shows up as additive density on
        // the shadow path so a sun-NEE ray hitting a plume gets the
        // correct Beer-Lambert dimming.
        dens += smoke_density_add(sp);
        // Fluid Phase 3 (#22): SPH particles also contribute to shadow
        // optical depth so an SPH-mode plume occludes the sun NEE leg
        // identically to the procedural plume. Returns 0 when no SPH
        // particles are bound -- no branch needed.
        dens += smoke_sph_density_add(sp);
        od += dens * dt;
    }
    return od;
}

// Phase 4 (#100): high-resolution stratified cloud transmittance for the
// cloud-march body's NEE light shadow rays. The legacy cloud_optical_depth
// above uses 4 uniform samples + 3 fbm octaves -- fine for surface-NEE
// transmittance() callers (light-on-ground god rays through clouds, where
// the cloud-shadow blur on the ground is masked by surface roughness) but
// too coarse for the in-cloud single-scatter NEE that drives god-ray
// volumetric shafts. Thin holes in the cloud field (a few-hundred-metre
// gap between cumulus puffs) get aliased away by 4-sample stride, and the
// shaft direction loses its silhouette.
//
// This variant:
//   * Stratifies up to 12 jittered samples across [t_in, t_out] so per-
//     pixel noise replaces grid aliasing in the cloud field's sub-100m
//     features.
//   * Uses 4 fbm octaves (vs 3) -- the higher freq drops aliased pixels
//     of dense-cloud cores back to plausible optical depth.
//   * Applies vol_density_scale uniformly so the user-facing
//     r_vol_density_scale knob multiplies the shadow transmittance the
//     SAME way it multiplies the primary cloud-march extinction.
//
// Caps stay identical to cloud_optical_depth (20km, layer-slab clip)
// so the two paths agree on which spans are sky vs. occlusion.
float cloud_optical_depth_stratified(float3 ro, float3 rd, float t_max,
                                     inout uint seed) {
    // Fluid Phase 1 (#136): also fire the stratified shadow march when
    // smoke is enabled, so a NEE sun ray through a smoke plume picks
    // up the correct Beer-Lambert occlusion. Without this, sunset
    // rim-lighting on smoke would over-brighten -- the smoke would
    // catch full sun even when it's directly between camera and the
    // sun. When clouds are off but smoke is on we drop the slab to
    // ground so smoke at chimney altitude actually gets sampled.
    // Smoke is "active" when either the procedural Phase 2 emitters
    // are bound (smoke_params.y gate + count) OR the Phase 3 SPH
    // particles are bound (smoke_sph_params.x > 0). The host's mode
    // selection in r_smoke_mode collapses one of the two to zero so
    // only one path contributes per frame; the OR keeps the gate
    // logic generic.
    bool smoke_active = (smoke_params.y > 0.0) && (smoke_params.x > 0.0);
    bool sph_active   = (smoke_sph_params.x > 0.0);
    smoke_active = smoke_active || sph_active;
    if (clouds_p1.w <= 0.0 && !smoke_active) return 0.0;
    float base_y = clouds_p1.x;
    float top_y  = clouds_p1.y;
    if (clouds_p1.w <= 0.0) {
        base_y = 0.0;
        top_y  = 2000.0;
    }
    float t_in, t_out;
    if (!cloudLayerInterval(ro, rd, base_y, top_y, t_max, t_in, t_out)) {
        return 0.0;
    }
    // Keep the NEAR window -- t_in is the entry nearest the ray origin.
    if (t_out - t_in > 20000.0) t_out = t_in + 20000.0;
    if (t_out <= t_in) return 0.0;
    const int Ns = 12;
    float dt = (t_out - t_in) / float(Ns);
    float dscale = max(vol_density_scale, 1e-3);
    float od = 0.0;
    // Per-sample stratified jitter so adjacent shadow rays get
    // decorrelated noise across the field rather than aliasing on the
    // same uniform-stride pattern.
    [unroll] for (int i = 0; i < Ns; ++i) {
        float xi = randf(seed);
        float t  = t_in + (float(i) + xi) * dt;
        float3 sp = ro + rd * t;
        float dens = cloud_density_at(sp, clouds_p1, clouds_p2, clouds_p3, clouds_p4, 4,
                                      planet_center_radius);
        // Fluid Phase 1 (#136): smoke contributions ride the same
        // shadow-ray transmittance so plumes self-shadow at sunset.
        dens += smoke_density_add(sp);
        // Fluid Phase 3 (#22): SPH particles too -- same self-
        // shadowing semantics. Returns 0 when no SPH particles bound.
        dens += smoke_sph_density_add(sp);
        od += dens * dt;
    }
    return od * dscale;
}

// Atmospheric transmittance along the straight ray segment
// [ro, ro + rd*t]. Gives cloud-march samples and primary-ray surface
// hits their aerial-perspective fade with distance regardless of sky
// mode -- an HDRI is the canonical sky source but does not model the air
// mass between the camera and a far cloud.
//
// THREE species since planetary P3 (#257), not two:
//
//   Rayleigh   molecular scattering, exponential in altitude, H = 8 km.
//              Wavelength^-4, so blue scatters ~5.7x more than red --
//              this is why the sky is blue and why a long slant path
//              turns the sun red.
//   Mie        aerosol, exponential, H = 1.2 km, near-grey. Scattering
//              AND absorption: the engine modelled no aerosol
//              absorption at all before #257, which made haze a pure
//              scatterer and so too bright at every optical depth.
//   Ozone      absorption only, and a stratospheric TENT peaking at
//              25 km, not an exponential. Completely absent before
//              #257. Its Chappuis band absorbs red-orange, so on the
//              long twilight slant path it removes exactly the part of
//              the spectrum that would otherwise make a post-sunset
//              zenith brown, and leaves it blue. That single term is
//              the cheapest real-physics win in the whole atmosphere
//              chain, and its omission was the largest error in it.
//
// The medium itself -- every coefficient, every scale height, every
// citation -- lives in PathTraceMath.slang's PtAtmoBody / ptAtmoEarth,
// parameterised by body. Nothing about Earth is spelled out in this
// file any more. That is deliberate and is the one structural decision
// in #257 whose cost multiplies if deferred: a second body is a second
// constructor rather than a search-and-replace across the four files
// that used to carry copies of the Rayleigh scale height.
//
// User scales, all real quantities with no compensating partner (that
// pairing is exactly what #257 retires):
//   vol_params.x  r_volumetric_density -- ABSOLUTE sea-level Mie
//                 scattering coefficient, m^-1.
//   clouds_p3.w   r_rayleigh -- multiple of the real Rayleigh column,
//                 1.0 = real Earth.
//   clouds_p4.w   r_ozone -- multiple of the real ozone column,
//                 1.0 = the US Standard Atmosphere 1976 column.
// All three at zero is a vacuum and is early-outed.
//
// GEOMETRY. When sun_extra.w > 0 the air is a shell around a sphere of
// that radius; the centre comes from planet_center_radius when the
// spherical world frame is active (#254) and otherwise defaults to
// (0, -R, 0), so the path tracer's flat y = 0 ground plane sits tangent
// to the body at the world origin -- the two coincide by construction.
// Altitude along the ray is the hoisted cancellation-free quadratic
// (#271), never a materialised position: at planetary radius the naive
// form carries ~0.5 m of noise into exp(-h/1200), which is 0.04% of the
// Mie scale height on every sample.
//
// The horizontal ray at sea level is the case that makes the shell
// worth its cost: spherical geometry gives it ~38 air masses, planar
// geometry gives it infinity.
//
// INTEGRATION. 8-step composite Simpson on the ray parameter, one rule
// for both geometries. Simpson is exact for cubics and both exponential
// species are smooth in a quadratic-in-s altitude, so 8 steps hold
// ~0.1%; ozone's tent has two corners Simpson cannot see and so carries
// a larger relative error, bounded and discussed at ptAtmoOpticalDepth.
//
// PRECONDITION: `rd` must be unit length. `dt = t / N` is treated as a
// step in metres, so a non-normalised direction corrupts BOTH the
// per-sample altitude and the integration measure. Every current call
// site passes a normalised direction. The function does NOT renormalise
// defensively; a future caller passing a world-space difference vector
// MUST normalise first.
//
// Below-surface origins are handled: altitudes clamp to 0 so the
// optical depth cannot go negative.
// The body this scene's air belongs to, with the engine's per-species
// user scales applied (planetary P3, #257). Earth's constants and their
// citations live in PathTraceMath.slang's ptAtmoEarth; nothing about
// Earth is spelled out here any more, which is the point -- the four
// hardcoded copies of the Rayleigh scale height and the two of the
// sigma triple are gone, and a second body is a second constructor
// rather than a search-and-replace across four files.
//
//   vol_params.x  r_volumetric_density -- ABSOLUTE sea-level Mie
//                 scattering coefficient in m^-1.
//   clouds_p3.w   r_rayleigh  -- multiple of the real Rayleigh column.
//   clouds_p4.w   r_ozone     -- multiple of the real ozone column.
PtAtmoBody atmosphereBody() {
    float planet_R = sun_extra.w;
    // The planar A/B (r_planet_radius 0) still needs a medium; only its
    // ALTITUDE function differs. Feed the constructor the real mean
    // radius there so the two halves of the A/B differ in exactly one
    // thing -- how h(s) is computed -- and not in what the air is made
    // of, which is what makes the pair a controlled comparison.
    float R = (planet_R > 0.0) ? planet_R : 6371008.8;
    return ptAtmoScale(ptAtmoEarth(R), clouds_p3.w, vol_params.x,
                       clouds_p4.w);
}

// Bilinear fetch of Hillaire 2020's multiple-scattering table (#280),
// mirrored by pt::atmo::SampleMultiScatter on the host and pinned against
// it in tests/pt_atmosphere_test.cpp.  Written out here rather than in
// PathTraceMath because passing a StructuredBuffer across a Slang module
// boundary is a construct this codebase has never lowered to SPIR-V, and
// #257 already spent a day on what module boundaries do to this compiler
// (its risk list, items 1 and 3).  The PARAMETERISATION does live in the
// module, as ptAtmoMsLutUv -- that half is model and belongs with the
// model.
//
// Returns radiance directly, WITH the solar irradiance already folded in
// by the builder, so the caller multiplies it by a scattering coefficient
// and by nothing else.
float3 ptAtmoMultiScatter(PtAtmoBody body, float r, float mu_s) {
    if (!ptMsLutReady()) return float3(0.0);
    float2 uv = ptAtmoMsLutUv(body, r, mu_s);
    float fx = uv.x * float(kPtMsLutWidth)  - 0.5;
    float fy = uv.y * float(kPtMsLutHeight) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    int xa = clamp(x0,     0, kPtMsLutWidth  - 1);
    int xb = clamp(x0 + 1, 0, kPtMsLutWidth  - 1);
    int ya = clamp(y0,     0, kPtMsLutHeight - 1);
    int yb = clamp(y0 + 1, 0, kPtMsLutHeight - 1);
    float3 c00 = atmo_ms_lut[uint(ya * kPtMsLutWidth + xa)].rgb;
    float3 c10 = atmo_ms_lut[uint(ya * kPtMsLutWidth + xb)].rgb;
    float3 c01 = atmo_ms_lut[uint(yb * kPtMsLutWidth + xa)].rgb;
    float3 c11 = atmo_ms_lut[uint(yb * kPtMsLutWidth + xb)].rgb;
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

float3 atmosphericTransmittance(float3 ro, float3 rd, float t) {
    if (t <= 0.0) return float3(1.0);
    PtAtmoBody body = atmosphereBody();
    // "No atmosphere" debug states stay reachable: all three scales at
    // zero means vacuum, and the early-out keeps that free rather than
    // integrating a zero.
    bool any_medium = (vol_params.x > 0.0) || (clouds_p3.w > 0.0)
                   || (clouds_p4.w > 0.0);
    if (!any_medium) return float3(1.0);

    float planet_R = sun_extra.w;
    if (planet_R <= 0.0) {
        // ---- Legacy planar path (r_planet_radius 0) ------------------
        // Kept as the A/B against the spherical shell, and for tiny-scene
        // tests where curvature is invisible.
        //
        // This used to be an ANALYTIC exponential integral, which was
        // possible because the medium was two exponentials. It is not
        // possible now: ozone is a tent, not an exponential, so no
        // closed form covers all three species. Rather than integrate
        // two species analytically and bolt the third on numerically --
        // which would make the planar and spherical halves of the A/B
        // differ in their QUADRATURE as well as in their geometry, and
        // so stop being a controlled comparison -- both halves now run
        // the same composite Simpson rule over the same medium, and
        // differ in exactly one line: how h(s) is computed.
        const int N = 8;
        float dt = t / float(N);
        float oneSixth = 1.0 / 6.0;
        float3 tau = float3(0.0);
        float3 sr_l, sm_l, st_prev;
        ptAtmoCoefficients(body, max(ro.y, 0.0), sr_l, sm_l, st_prev);
        for (int i = 0; i < N; ++i) {
            float s_mid   = (float(i) + 0.5) * dt;
            float s_right = (float(i) + 1.0) * dt;
            float3 sr_m, sm_m, st_mid;
            float3 sr_r, sm_r, st_right;
            ptAtmoCoefficients(body, max(ro.y + rd.y * s_mid,   0.0),
                               sr_m, sm_m, st_mid);
            ptAtmoCoefficients(body, max(ro.y + rd.y * s_right, 0.0),
                               sr_r, sm_r, st_right);
            tau += dt * oneSixth * (st_prev + 4.0 * st_mid + st_right);
            st_prev = st_right;
        }
        return exp(-tau);
    }

    // ---- Spherical shell (issue #51, three species since #257) -------
    // Shell centre. The planar world frame has no planet centre, so the
    // integral keeps #51's convention: put it at (0, -R, 0), which makes
    // the engine's y = 0 ground plane tangent to the sphere at the world
    // origin. The spherical frame pushes the real centre and the engine
    // places it at exactly that same point, so the two agree by
    // construction -- but reading it from the push means this integral
    // and localUp() can never drift into different frames. `p.y - pc.y`
    // with pc.y = -R is `p.y + R` bit-for-bit (IEEE negation is exact),
    // and `p.x - 0` is `p.x`.
    float3 pc = (planet_center_radius.w > 0.0) ? planet_center_radius.xyz
                                               : float3(0.0, -planet_R, 0.0);
    return exp(-ptAtmoOpticalDepth(body, ro, rd, pc, t, 8));
}

// Transmittance of the atmosphere along the SUN ray from `p` -- the
// colour and brightness sunlight actually has by the time it arrives
// (planetary P3, #257).
//
// WHAT THIS REPLACES. The engine reddens sunlight with
//     am    = 1 / max(sin(elev), 0.04)
//     trans = exp(-0.30 * am)
//     col   = lerp((1.00, 0.97, 0.92), (1.55, 0.55, 0.20),
//                  saturate((am - 1) * 0.18))
// -- a grey exponential in a thin-slab airmass, times a hand-authored
// lerp between two invented RGB endpoints. Four fabricated constants
// standing in for a quantity the engine can now compute: the real
// slant-path optical depth through Rayleigh, aerosol AND ozone. This is
// the single largest remaining unsourced constant in the sky chain, and
// it is also the ONLY path on which ozone is visible in a rendered
// frame -- a camera-to-surface ray never climbs anywhere near 25 km, so
// the aerial-perspective integral cannot see the layer, while the
// SUNLIGHT crosses it at a grazing angle every twilight.
//
// OFF BY DEFAULT (sun_extra2.y). Switching sunlight from a tuned curve
// to a physical one changes the brightness of every lit surface in
// every scene -- measured at noon, the physical sun is about 30%
// brighter and differently tinted than exp(-0.30*am) -- so it is a
// deliberate opt-in exactly like r_planet_spherical_frame, and the
// default flip belongs with the radiometric unit rebase that the rest
// of #257's parallel-golden machinery is for. What lands here is the
// capability and its proof, not the flip.
//
// GEOMETRY. Clip the ray to the top of the shell and integrate. A sun
// below the local horizon is occluded by the body itself and returns
// zero -- which is the physically correct statement of "night" and
// needs no elevation threshold. In the planar A/B (r_planet_radius 0)
// there is no body to occlude and no shell to clip to, so the path
// length is the slant column of a 100 km slab, 1/sin(elev), which
// diverges at the horizon: that divergence IS the planar model's
// defect and the A/B exists to show it.
float3 sunSlantTransmittance(float3 p, float3 sun_dir) {
    PtAtmoBody body = atmosphereBody();
    float planet_R = sun_extra.w;
    if (planet_R <= 0.0) {
        float sy = max(sun_dir.y, 1e-3);
        float t = (100000.0 - p.y) / sy;
        if (!(t > 0.0)) return float3(1.0);
        return atmosphericTransmittance(p, sun_dir, t);
    }
    float3 pc = (planet_center_radius.w > 0.0) ? planet_center_radius.xyz
                                               : float3(0.0, -planet_R, 0.0);
    // Occluded by the body: the ray leaves p, descends, and re-enters
    // the surface before it can reach space.
    float g0, g1;
    if (ptSphereRoots(p, sun_dir, pc, planet_R, g0, g1) && g0 > 0.0) {
        return float3(0.0);
    }
    float s0, s1;
    if (!ptSphereRoots(p, sun_dir, pc, body.top_radius, s0, s1)) {
        return float3(1.0);          // already outside the shell
    }
    float t = max(s1, 0.0);
    if (!(t > 0.0)) return float3(1.0);
    return exp(-ptAtmoOpticalDepth(body, p, sun_dir, pc, t, 8));
}

// Solar IRRADIANCE arriving at `p`, in W/m^2, on a surface facing the sun.
// `sun_elev` is sin(elevation) at p, used only by the legacy curve.
//
// THE NAME IS THE POINT (#280).  This function was called sunRadianceAt
// and its comment said "80 is the unattenuated solar constant" -- but the
// solar constant is an IRRADIANCE and solar radiance is 14 700x larger, so
// the name and the comment named different quantities and the caller could
// not tell which the code meant.  The arithmetic was never ambiguous: every
// call site multiplies this by a BRDF and a cosine with no solid angle
// anywhere, which closes dimensionally only for an irradiance, and
// sunDiscPhysical divides it by Omega to recover a radiance.  So the code
// was right and the label was wrong.  The audit, with all three proofs, is
// in src/renderer/Atmosphere.h section 1.
//
// The 80 is now kPtSolarIrradiance -- 1360.8 W/m^2 (Kopp & Lean 2011) split
// R:G:B by the solar spectrum at this model's own three sampling
// wavelengths (Bruneton 2017).  See PathTraceMath.slang.
//
// The two branches are the SAME quantity computed two ways, so every
// site that needs sunlight can call this and none of them can drift.
// Before #257 the expression was copied at four sites with three
// slightly different tunings between them.
float3 sunIrradianceAt(float3 p, float3 sun_dir, float sun_elev) {
    if (sun_extra2.y > 0.0) {
        return kPtSolarIrradiance * sunSlantTransmittance(p, sun_dir);
    }
    // Legacy tuned curve (r_sun_physical_transmittance 0), kept as the A/B
    // against the real slant path.  #280 flipped the DEFAULT to the
    // physical branch -- #257 measured that flip as ~30% brighter and
    // differently tinted at 60 deg and explicitly handed it to the unit
    // rebase, which is this change.
    //
    // Its scale is the GREY mean of the irradiance rather than the triple,
    // because this branch carries its own hand-authored colour ramp and
    // multiplying a painted tint by a physical tint would tint twice.  One
    // number, kPtLegacySkyScale * 80, so the curve keeps exactly the shape
    // its author gave it at the new absolute level.
    float am  = 1.0 / max(sun_elev, 0.04);
    float tau = 0.30;             // tunable haze
    float trans_atm = exp(-tau * am);
    float warmth = saturate((am - 1.0) * 0.18);
    float3 sun_col = lerp(float3(1.00, 0.97, 0.92),
                          float3(1.55, 0.55, 0.20),
                          warmth);
    return sun_col * (80.0 * kPtLegacySkyScale * trans_atm);
}

// --- Planetary P6 (#260): the sky that survives leaving ---------------------
//
// r_sky_mode physical.  Sky radiance is INTEGRATED along the view ray
// rather than painted, and that single property is what makes the climb
// from the surface to orbit continuous.
//
// WHY A FIFTH MODE AND NOT A FIX TO procSky
//
// procSky and hosekSky are analytic sky DOMES: functions of view
// direction with the observer implicitly on the ground.  They are
// accelerators for a ground observer, and good ones -- hosek in
// particular is the genuine ArHosekSkyModel dataset.  Neither can leave.
// procSky's zenith is the literal float3(0.32, 0.55, 0.92) at eye height
// and the same literal at 400 km; hosek clamps cos_theta to [0, 1] and
// fades its horizon, so it has no representation for "most of the sphere
// is vacuum" at all.  Attempts to rescue them are all the same shape --
// fade the painted dome out on altitude -- and a fade is a tuned
// constant standing exactly where the discontinuity is.
//
// The physical mode has no accelerator to swap out, so there is no
// crossover to hide.  At 1.7 m the chord through the shell is 100 km of
// dense air and the integral is blue; at 400 km looking up the chord is
// empty and the integral is zero, so the sky is black.  No fade, no
// threshold, no cvar: it falls out of the geometry.  The roadmap
// specifies a sky-view LUT gated on camera-inside-shell with a 10 km
// cross-fade; direct evaluation everywhere is the same design with the
// accelerator (and therefore the cross-fade, and therefore the tuning)
// removed, which this engine can afford because it never rasterizes and
// already marches per pixel.
//
// THE SAMPLE DISTRIBUTION IS ALTITUDE-INDEPENDENT, WHICH IS THE WHOLE
// CONTINUITY ARGUMENT
//
// Uniform-in-distance stepping over [0, t] is catastrophic from orbit --
// a nadir ray from 400 km is 400 km long and 90% of its optical depth is
// in the last 20 km.  Stepping over the SHELL-CLIPPED interval instead
// removes that failure without any stratification at all, because the
// clipped interval always spans the same air whatever the camera's
// altitude: a ground camera looking up and an orbital camera looking
// down integrate the same 100 km column with the same N samples in the
// same places.  The quadrature error is therefore a function of the
// geometry of the ray and not of where the camera is, and an error that
// does not move as the camera climbs cannot produce a pop.
//
// Samples are jittered (`(i + xi) * dt`), which makes this an unbiased
// estimator of the integral rather than a fixed-point rule, so the
// accumulator converges the residual quadrature error away over frames
// instead of baking it in.
//
// WHAT IS IN THE INTEGRAL
//
//   L = integral over the shell chord of
//         T(camera -> s) * [ sigma_s_R(h) P_R(mu) + sigma_s_M(h) P_CS(mu, g) ]
//                        * L_sun * T(s -> sun)
//
// Three species from the body (#257), both real phase functions, and the
// sun term is the same sunSlantTransmittance the NEE site and the cloud
// march use -- so the light the sky is made of and the light the ground
// is lit by cannot disagree.  T(s -> sun) returning zero when the body
// occludes the sun IS the night side, and the terminator is the graded
// region where it is partial: no elevation threshold anywhere.
//
// WHAT IS NOT IN IT, STATED RATHER THAN HIDDEN
//
//   * (RESOLVED in #280) MULTIPLE SCATTERING.  Was single-scatter only,
//     which made the sky darker than reality -- most visibly at the
//     zenith on a clear day and through twilight, where second-order
//     Rayleigh is a large fraction of the total, and which is exactly
//     what the retired r_rayleigh = 30 was standing in for.  Hillaire
//     2020 section 4's 32x32 (r, mu_s) table now supplies it, built on
//     the host into atmo_ms_lut.  Measured contribution at the zenith:
//     +27% at noon, +63% at 4 degrees below the horizon.
//   * (RESOLVED in #280) GROUND-REFLECTED LIGHT.  The multi-scatter
//     table's build carries the planet-albedo bounce (r_sky_ground_albedo)
//     into the isotropic term, so the ground does light the air above it
//     -- at the order the table models, which is second and above.  The
//     FIRST-order ground bounce along this particular view ray is still
//     absent.
//   * SCENE OCCLUSION of the sky column.  Terrain and clouds do not
//     shadow this march; the analytic BODY does, via
//     sunSlantTransmittance, and at planetary scale that is the term
//     that matters.  The primary-HIT aerial-perspective march still uses
//     transmittance(), so god rays through cloud gaps are unaffected.
//
// PRECONDITION: `rd` is unit length.  It is both the integration measure
// and the s^2 coefficient of the hoisted altitude quadratic.
float3 skyPhysical(float3 ro, float3 rd, float t_max, int steps,
                   inout uint seed) {
    float planet_R = sun_extra.w;
    // The physical sky is a statement about a body.  Without one there is
    // no shell to clip to and no horizon to occlude the sun, so the mode
    // has nothing to integrate; the host refuses to select it in that
    // state and this is the shader-side guard for the same condition.
    if (planet_R <= 0.0) return float3(0.0);
    PtAtmoBody body = atmosphereBody();
    float3 pc = (planet_center_radius.w > 0.0) ? planet_center_radius.xyz
                                               : float3(0.0, -planet_R, 0.0);
    float t_in, t_out;
    // r_inner = 0 so this is the near piece of the ray inside top_radius,
    // correct for an origin below, inside or above the shell alike.
    if (!ptRayShell(ro, rd, pc, 0.0, body.top_radius, t_max, t_in, t_out)) {
        return float3(0.0);        // the ray never touches air -- space
    }
    // The body is opaque: stop the column where the ground starts.
    float g0, g1;
    if (ptSphereRoots(ro, rd, pc, planet_R, g0, g1) && g0 > 0.0) {
        t_out = min(t_out, g0);
    }
    if (!(t_out > t_in)) return float3(0.0);

    float3 sun_dir = sun_and_mode.xyz;
    // Scattering angle.  Incident light propagates along -sun_dir and
    // leaves toward -rd, so the phase cosine is dot(rd, sun_dir) -- the
    // same convention the in-scatter march uses.
    float mu   = dot(rd, sun_dir);
    float ph_r = ptPhaseRayleigh(mu);
    // The aerosol asymmetry comes from r_volumetric_anisotropy
    // (vol_params.y, default 0.7) and NOT from the body's own mie_g, which
    // ptAtmoEarth sets to the Cornette-Shanks 0.8.  Deliberate: the
    // aerial-perspective march next door reads the same cvar, and the two
    // meet at the limb, so a disagreement here would be a seam.  Honouring
    // the user's knob in both is worth more than honouring the body's
    // constant in one of them.  Unifying them is a change to the AP march
    // as well, which moves clouds_godrays, so it belongs with that
    // regeneration and not here.
    float ph_m = ptPhaseCornetteShanks(mu, vol_params.y);

    int   N  = max(steps, 1);
    float dt = (t_out - t_in) / float(N);
    float xi = randf(seed);
    // Hoisted cancellation-free altitude (#271): the quadratic is
    // accumulated once and each sample costs two fma, instead of
    // materialising p(s) and subtracting two ~6.4e6 quantities.
    PtRayAltitude alt = ptRayAltitudeBegin(ro, rd, pc, body.ground_radius);
    float3 trans = float3(1.0);
    float3 acc   = float3(0.0);
    for (int i = 0; i < N; ++i) {
        float s = t_in + (float(i) + xi) * dt;
        if (s >= t_out) break;
        float h = max(ptRayAltitudeAt(alt, s), 0.0);
        float3 sigma_s_ray, sigma_s_mie, sigma_t;
        ptAtmoCoefficients(body, h, sigma_s_ray, sigma_s_mie, sigma_t);
        // Sunlight AT THIS SAMPLE.  Zero when the body is between the
        // sample and the sun, which is what "night" means and needs no
        // elevation gate; reddened by the real slant path, which is what
        // the terminator and orbital sunrise are made of.
        float3 sp     = ro + rd * s;
        float3 sun_at = kPtSolarIrradiance * sunSlantTransmittance(sp, sun_dir);
        if (any(sun_at > float3(0.0))) {
            acc += trans * (sigma_s_ray * ph_r + sigma_s_mie * ph_m)
                 * sun_at * dt;
        }
        // Second and higher orders (#280).  Isotropic, so it carries no
        // phase function; the table already holds radiance with the solar
        // irradiance folded in, so it multiplies the scattering
        // coefficient and nothing else.  It is NOT inside the
        // `any(sun_at > 0)` gate: a sample in the planet's shadow at
        // twilight still receives multiply-scattered light from the lit
        // air around it, and that is what the blue hour is made of.
        float3 up_at   = normalize(sp - pc);
        float3 ms_here = ptAtmoMultiScatter(body, length(sp - pc),
                                            dot(up_at, sun_dir));
        acc += trans * (sigma_s_ray + sigma_s_mie) * ms_here * dt;
        trans *= exp(-sigma_t * dt);
    }
    // r_volumetric_intensity, the same artistic gain the in-scatter march
    // applies, so the sky and the aerial perspective under it move
    // together instead of drifting apart when a user turns it.
    return acc * vol_params.z;
}

// The sun's own disc in the physical mode.
//
// Two things change from sunDisc().  The `sun_elev < -0.10` cutoff goes:
// whether the sun is visible is whether the BODY is in the way, which
// sunSlantTransmittance already answers exactly, and from outside the
// atmosphere there is no elevation at which the sun should vanish.  And
// the exp(-0.30 / max(sun_elev, 0.04)) attenuation with its hand-authored
// warm/cool endpoint lerp goes: the disc's colour is the slant-path
// transmittance, so the sun reddens through a real air mass on the way
// down and is pure white from orbit, and orbital sunrise -- the sun
// climbing out from behind the limb through a real grazing column --
// costs nothing extra because it is the same expression.
//
// The halo goes too, and that is a deletion rather than a loss: the
// bright bloom around a low sun is forward Mie scattering, which
// skyPhysical now computes properly through the Cornette-Shanks phase at
// g = 0.8.  Drawing the painted halo as well would double-count it.
//
// DISC RADIANCE IS DERIVED, NOT TUNED.  sunIrradianceAt returns E * T as
// the solar irradiance, now in real W/m^2 (#280).  A disc of
// solid angle Omega carrying irradiance E has radiance E / Omega, so the
// rendered disc is E * T / Omega with Omega = 2 pi (1 - cos r) for the
// rendered half-angle r.  Taking Omega from the RENDERED radius rather
// than from the sun's true 0.2666 deg makes r_sun_size energy-conserving:
// a bigger disc is proportionally dimmer and puts the same flux on the
// scene, which is the property that stops the disc size from being a
// brightness knob in disguise.
float3 sunDiscPhysical(float3 ro, float3 rd, float3 sun_dir) {
    float3 T = sunSlantTransmittance(ro, sun_dir);
    if (!any(T > float3(0.0))) return float3(0.0);   // body in the way
    float ang = dot(rd, sun_dir);
    // Same angular geometry as sunDisc: r_sun_size times the
    // perihelion/aphelion ratio from the ephemeris (sun_extra.y).
    float sun_mult = max(sun_extra.x * sun_extra.y, 0.1);
    float r_deg    = 0.55 * sun_mult;
    float r_rad    = r_deg * 0.0174533;
    float kCosR    = cos(r_rad);
    if (ang <= kCosR) return float3(0.0);

    // Saemundsson horizon flatten (r_sun_horizon_flatten, sun_extra.z).
    // Real refraction, kept -- but its trigger is the APPARENT elevation
    // at the camera, which the host now derives from the camera's own
    // local up rather than from the global sun.y.
    float vflat = sun_extra.z;
    if (vflat < 0.999) {
        float3 world_up    = localUp(ro);
        float3 local_right = normalize(cross(world_up, sun_dir));
        float3 local_up    = cross(sun_dir, local_right);
        float3 delta       = rd - sun_dir * ang;
        float  h_off       = dot(delta, local_right);
        float  v_off       = dot(delta, local_up);
        float  r_v         = r_rad * vflat;
        float  ellipse     = (h_off * h_off) / (r_rad * r_rad)
                           + (v_off * v_off) / (r_v * r_v);
        if (ellipse > 1.0) return float3(0.0);
    }

    float omega = 6.28318530718 * (1.0 - kCosR);   // sr, rendered disc
    float bright = max(sun_extra2.x, 0.0);         // r_sun_disc_brightness
    return T * (kPtSolarIrradiance / max(omega, 1.0e-12)) * bright;
}
// --- end Planetary P6 physical sky ------------------------------------------

// `ro` is the ray's ORIGIN, and it is load-bearing only in mode 4
// (planetary P6, #260).  procSky documents an approximation it has always
// carried: it reads the CAMERA's altitude out of pos_fovtan even when a
// secondary bounce called it, so a puddle at ground level under a 10 km
// camera sees a 10 km horizon.  From orbit that error is 400 km.  The
// physical mode takes the origin per call, so the approximation is simply
// absent there -- every call site already has the right origin in scope
// (the bounce's `ro`, or the surface point), it was never plumbed.  Modes
// 0-3 ignore the parameter and are unchanged.
float3 skyColor(float3 ro, float3 rd, inout uint seed) {
    float3 base;
    uint mode = uint(sun_and_mode.w);
    // --- Planetary P6 (#260): mode == 4 -> physical, marched sky --------
    // The only mode that survives leaving the atmosphere.  There is no
    // painted base at all: the integral IS the sky, so it goes black
    // outside the shell without a fade.  Celestials ride on top exactly
    // as they do in the procedural and hosek modes, except that the sun
    // disc is the physical one.
    if (mode == 4u) {
        return skyPhysical(ro, rd, 1.0e30, int(max(vol_params.w, 1.0)), seed)
             + starsOnly(rd, sun_and_mode.xyz)
             + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDiscPhysical(ro, rd, sun_and_mode.xyz);
    }
    // --- Wave 9 hosek-sky ---
    // mode == 3 -> Hosek-Wilkie physically-based sky. Routed like the
    // procedural path (mode 2): the analytic dome drives both the
    // visible sky AND the diffuse-skylight ambient sampled at secondary
    // bounces (skyColorBase(hit_pt, nf, seed) in the Lambert NEE block),
    // and the same
    // sun/moon disc overlays + stars are composited on top.
    if (mode == 3u) {
        base = hosekSky(rd, sun_and_mode.xyz);
        return base + moonDisc(rd, moon_dir_phase, sun_and_mode.xyz)
             + sunDisc(rd, sun_and_mode.xyz);
    }
    // --- end Wave 9 hosek-sky ---
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

// Sky WITHOUT the sun/moon disc + halo overlays. The diffuse-ambient
// estimator in the Lambert NEE block must not see the celestial discs:
// sun and moon direct light are integrated by their own NEE blocks
// (with real shadow rays), so an ambient tap that includes disc/halo
// radiance double-counts them with NO visibility test -- sun-facing
// slopes at sunset read visibly too bright and receive unshadowed
// "ambient" sun. Mirrors the composite-subtraction at the primary-miss
// branch, which is the single source of truth for which terms count as
// celestial overlays.
float3 skyColorBase(float3 ro, float3 rd, inout uint seed) {
    uint mode = uint(sun_and_mode.w);
    if (mode == 4u) {
        // Physical mode (#260) carries no inline celestial term at all,
        // so the base IS the march.  It runs at a QUARTER of the primary
        // budget with a floor of 4: this tap is one sample of a diffuse
        // hemispherical integral that the path tracer is already
        // estimating stochastically, so its own quadrature error is
        // averaged over the same accumulation the ambient term is -- the
        // visible sky, which is not averaged over directions, is what
        // needs the full budget.  A cost choice, stated as one.
        return skyPhysical(ro, rd, 1.0e30,
                           max(int(vol_params.w) / 4, 4), seed);
    }
    if (mode == 3u) {
        // Hosek-Wilkie dome radiance carries no inline disc; skyColor
        // adds sunDisc/moonDisc on top, which is exactly what we skip.
        return hosekSky(rd, sun_and_mode.xyz);
    }
    if (mode == 2u) {
        // procSky bakes its own inline sun disc + halo -- subtract it
        // with the same helper the celestial composite uses.
        return procSky(rd, sun_and_mode.xyz)
             - procSkyInlineSunDiscAndHalo(rd, sun_and_mode.xyz);
    }
    if (env_map_present != 0u) {
        // HDRI path adds no procedural overlays; the map IS the base.
        return skyColor(ro, rd, seed);
    }
    float t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
    return lerp(float3(0.50, 0.55, 0.62), float3(1.0, 1.0, 1.0), t);
}

// --- Planetary P5 (#259): the sky a lobe that already NEE'd the sun sees ---
//
// Full sky budget, minus exactly the sun DISC that skyColor injects for the
// current mode. Not skyColorBase: that also drops the moon, the stars and
// (in physical mode) three quarters of the march budget, and a water
// reflection is the one place the sky is the subject rather than an ambient
// tap.
//
// WHY IT IS EXACT IN PHYSICAL MODE. sunDiscPhysical returns
// T * E / omega and sunIrradianceAt returns E * T, so the disc's radiance
// times its solid angle IS the irradiance the NEE integrates -- the two
// estimators partition the same light with no seam and no scale factor to
// agree on. In procedural and Hosek modes sunDisc bundles a 2-8 degree
// atmospheric HALO with the disc, so subtracting it also removes the
// reflected halo: a small under-count in those modes, in the conservative
// direction, and the alternative is a disc-only variant of a function three
// other call sites share.
float3 skyColorNoSunDisc(float3 ro, float3 rd, inout uint seed) {
    uint mode = uint(sun_and_mode.w);
    float3 s = skyColor(ro, rd, seed);
    if (mode == 4u) {
        s -= sunDiscPhysical(ro, rd, sun_and_mode.xyz);
    } else if (mode == 3u) {
        s -= sunDisc(rd, sun_and_mode.xyz);
    } else if (mode == 2u) {
        s -= sunDisc(rd, sun_and_mode.xyz)
           + procSkyInlineSunDiscAndHalo(rd, sun_and_mode.xyz);
    }
    return max(s, float3(0.0));
}

// Up to kMaxRefract dielectric bounces; beyond that the ray gives up
// (returns 0) so a pathological all-glass scene can't loop forever.
// TIR also returns 0 (light can't pass that interface).
//
// Motion blur (#85): shutter_t01 is the shutter-time sample the calling
// path-trace iteration used for its primary ray. Threading it through
// to the shadow-ray scene query keeps the shadow consistent with the
// rendered surface -- a sphere that primary-hit at the prev-frame end
// of its swept volume should cast a shadow against that same position.
// `cone` is the ray cone at `ro` (issue #256). It travels with the ray
// through the refraction chain below -- each segment re-anchors it at the
// new origin -- so the near cut stays correct after the ray has crossed a
// dielectric and moved a long way from where it started.
float3 transmittance(float3 ro, float3 rd, float t_max, float shutter_t01,
                     PtRayCone cone, uint start_medium = 0u) {
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
    PtRayCone c = cone;
#if PT_WATER_ENABLED
    // Per-segment Beer's-law bookkeeping. Callers whose shadow ray
    // starts underwater (Lambert NEE from submerged geometry) pass the
    // caller's own medium stack; each transmissive crossing below
    // pushes or pops it. The camera/bounce path applies EXACT
    // per-segment absorption, and the old fixed "nominal 1.5 m per
    // crossing" shortcut here let direct sun light reach a 10 m sea
    // floor at near-full strength while the viewing path was dimmed by
    // the true exp(-absorb * depth) -- deep water looked sun-lit from
    // above but murky to the eye.
    //
    // Planetary P7 (#261): this used to be a single water/not-water bit
    // that flipped only for MAT_WATER. It is now the full stack, so a
    // shadow ray that crosses glass on its way up through water stops
    // attenuating with water absorption while it is inside the glass.
    uint sr_medium = start_medium;
#endif
    for (int i = 0; i <= kMaxRefract; ++i) {
        HitInfo h = traceScene(o, d, shutter_t01, c);
#if PT_WATER_ENABLED
        // Absorb along however much of this segment lies in water --
        // up to the hit, or up to t_max on the final unoccluded leg.
        if (ptMediumIsWater(sr_medium)) {
            float seg = h.hit ? min(h.t, t_max) : t_max;
            // #305: c_eff, not the bare absorption. This function carries
            // ILLUMINATION through the water -- the sun to a sea floor, the
            // sun to an in-scatter sample -- and the coefficient that
            // attenuates a diffuse downwelling stream is a + b_b, not a
            // alone (which loses the backscattered light entirely) and not
            // a + b (which loses the forward-scattered light that is still
            // travelling downward). Absorption alone was the pre-#305
            // spelling; it is a + b_b now, which is the same law the
            // in-scatter march and the water-leaving closed form use, and
            // that shared law is what makes a camera just above and just
            // below the surface agree instead of disagreeing by 11% in the
            // blue. See ptWaterEffAtten.
            t *= exp(-ptWaterEffAtten() * max(seg, 0.0));
        }
#endif
        if (!h.hit || h.t >= t_max) return t * cloud_t;     // unoccluded
        // Water (#134) joins dielectric on the transmissive-blocker list.
        // Both refract the shadow ray; only water folds Beer's-law
        // absorption into the transmittance via water_params0.xyz.
        //
        // PT_WATER_ENABLED gate: when OFF the MAT_WATER branch is
        // compiled out -- water is treated like any opaque material
        // (returns 0 contribution from this shadow ray). Saves
        // megakernel registers + DAG width in scenes that don't use
        // water (the common case).
#if PT_WATER_ENABLED
        bool is_transparent = (h.mat == MAT_DIELECTRIC) || (h.mat == MAT_WATER);
#else
        bool is_transparent = (h.mat == MAT_DIELECTRIC);
#endif
        if (!is_transparent)         return float3(0.0);    // opaque blocker
        if (i == kMaxRefract)        return float3(0.0);    // budget hit

        // Refract through the dielectric / water face. Same Snell +
        // Fresnel logic as the path-tracer's dielectric BRDF, but we
        // keep only the transmitted lobe (the reflected lobe doesn't
        // reach the light along this path). Fresnel is folded into
        // the transmittance so glancing angles correctly attenuate.
        float ndi   = dot(h.normal, d);
        float3 nf   = (ndi < 0.0) ? h.normal : -h.normal;
#if PT_WATER_ENABLED
        float interface_ior = (h.mat == MAT_WATER)
            ? clamp(h.ior > 0.0 ? h.ior : water_params0.w, 1.0, 2.4)
            : h.ior;
#else
        float interface_ior = h.ior;
#endif
        // Planetary P7 (#261) DELIBERATELY LEAVES THIS AT 1.0. Reading the
        // outside index off the medium stack -- which the stack now makes
        // possible -- was implemented, MEASURED and reverted, and the
        // measurement is the reason. See the matching note in the MAT_WATER
        // branch below.
        float ior_a = (ndi < 0.0) ? 1.0           : interface_ior;
        float ior_b = (ndi < 0.0) ? interface_ior : 1.0;
        float eta   = ior_a / ior_b;
        float cos_i = abs(dot(nf, -d));
#if PT_WATER_ENABLED
        float r0    = (h.mat == MAT_WATER)
            ? 0.02                             // accurate F0 for water at 0 deg
            : ((ior_a - ior_b) / (ior_a + ior_b)) * ((ior_a - ior_b) / (ior_a + ior_b));
#else
        float r0    = ((ior_a - ior_b) / (ior_a + ior_b)) * ((ior_a - ior_b) / (ior_a + ior_b));
#endif
        float fr    = fresnelSchlick(cos_i, r0);   // reflected fraction
        float3 refracted = refract(d, nf, eta);
        if (length(refracted) < 1e-3) return float3(0.0);   // TIR
        // Transmitted fraction times Beer-ish tint.  Water uses its own
        // per-channel exp(-absorption * nominal_seg) instead of h.albedo
        // (which would be neutral 1,1,1 for a water plane).
#if PT_WATER_ENABLED
        if (h.mat == MAT_WATER) {
            // Fresnel-transmitted fraction only. Absorption is applied
            // per-SEGMENT at the top of the loop (matching the exact
            // Beer's law the camera path uses), so crossing the
            // interface just flips which side of the surface we're on.
            t *= (1.0 - fr);
            sr_medium = ptMediumCross(sr_medium, MEDIUM_WATER);
        } else {
            t *= (1.0 - fr) * h.albedo;
            // Planetary P7 (#261): glass now pushes and pops too, so a
            // shadow ray inside a submerged bubble stops picking up water
            // absorption. Above water this only ever moves the stack
            // between "empty" and "one dielectric", and nothing reads the
            // stack in that case.
            sr_medium = ptMediumCross(sr_medium, MEDIUM_DIELECTRIC);
        }
#else
        t *= (1.0 - fr) * h.albedo;
#endif
        // Step past the interface, continue in the refracted direction.
        // The step past is the near cut AT THE INTERFACE, not an absolute
        // millimetre (#256): the ray has to clear the surface it just
        // crossed by the same distance that counts as "the same point"
        // there, which is a footprint at range and a few ULPs up close.
        PtRayCone c_hit = ptConeAdvance(c, h.t);
        float advance = h.t + ptRayTMin(o + d * h.t, c_hit);
        o = o + d * advance;
        d = normalize(refracted);
        c = ptConeAdvance(c, advance);
        t_max = max(t_max - advance, 0.0);
    }
    return float3(0.0);
}

// Convenience predicate around transmittance() for the metal /
// dielectric branches that just want the legacy boolean answer.
// Motion blur (#85): forwards shutter_t01 to keep shadow geometry
// consistent with the primary ray that originated the test.
bool occluded(float3 ro, float3 rd, float t_max, float shutter_t01,
              PtRayCone cone) {
    return all(transmittance(ro, rd, t_max, shutter_t01, cone) <= float3(0.0));
}

// --- Water Phase 1 (#134) --------------------------------------------------
// Gated by PT_WATER_ENABLED: when OFF the entire wave-normal helper and
// the MAT_WATER BRDF branch below compile out -- a Mac-release build of
// the megakernel without water support drops ~3 ms / frame off the
// pathtrace dispatch (8 noise evals + tangent-frame algebra per hit are
// expensive in the per-thread register pressure they produce, regardless
// of whether any actual MAT_WATER hit happens).  See PR #164.
#if PT_WATER_ENABLED
// Build a perturbed surface normal for MAT_WATER hits.  The geometry stays
// a flat analytic plane (h.normal is the macro normal); only the SHADING
// normal is animated.  We approximate the gradient of a 2-octave value-
// noise scalar field over world-space tangent coords with two finite-
// difference samples per axis -- cheap (4 noise evals per hit), no curl
// machinery needed (true curl-noise is solenoidal and unsigned, which is
// overkill for a shading-only perturbation).
//
// Inputs:
//   world_p : surface hit point in world space (xz used; y irrelevant for
//             a horizontal water plane).
//   macro_n : geometric normal (unit, world space).
//
// Output: unit normal in world space, perturbed in the plane orthogonal
// to macro_n.  Amplitude / frequency / time drive comes from water_params1.
// The macro normal is preserved verbatim when amplitude == 0 (regression-
// safe path for goldens that pin r_water_wave_amplitude 0).
float3 waterShadingNormal(float3 world_p, float3 macro_n) {
    float scale = water_params1.x;
    float amp   = water_params1.y;
    float speed = water_params1.z;
    float t     = water_params1.w * speed;
    if (amp <= 0.0 || scale <= 0.0) return macro_n;

    // 2-octave fbm-ish sum.  We sample two displaced copies of the noise
    // field (each animated in a different direction) and finite-difference
    // them in both x and z.  Mixing two drift directions decorrelates the
    // axis-aligned ripple lines that a single advected octave produces.
    // Frequencies match cloud_fbm's 2.02x multiplier so we don't introduce
    // axis-aligned aliasing artefacts.
    const float eps = 0.15;
    float2 p_xz = float2(world_p.x, world_p.z) * scale;
    // Octave A: drift along +x at unit speed.
    float2 drift_a = float2(0.13, 0.07) * t;
    // Octave B: drift along ~diagonal at 2.02x freq.
    float2 drift_b = float2(-0.09, 0.11) * t;

    float n_px = cloud_value_noise(float3(p_xz.x + eps, 0.0, p_xz.y) + float3(drift_a, 0).xzy)
               + 0.5 * cloud_value_noise(float3(p_xz.x * 2.02 + eps, 0.0, p_xz.y * 2.02) + float3(drift_b, 0).xzy);
    float n_nx = cloud_value_noise(float3(p_xz.x - eps, 0.0, p_xz.y) + float3(drift_a, 0).xzy)
               + 0.5 * cloud_value_noise(float3(p_xz.x * 2.02 - eps, 0.0, p_xz.y * 2.02) + float3(drift_b, 0).xzy);
    float n_pz = cloud_value_noise(float3(p_xz.x, 0.0, p_xz.y + eps) + float3(drift_a, 0).xzy)
               + 0.5 * cloud_value_noise(float3(p_xz.x * 2.02, 0.0, p_xz.y * 2.02 + eps) + float3(drift_b, 0).xzy);
    float n_nz = cloud_value_noise(float3(p_xz.x, 0.0, p_xz.y - eps) + float3(drift_a, 0).xzy)
               + 0.5 * cloud_value_noise(float3(p_xz.x * 2.02, 0.0, p_xz.y * 2.02 - eps) + float3(drift_b, 0).xzy);
    float dh_dx = (n_px - n_nx) / (2.0 * eps);
    float dh_dz = (n_pz - n_nz) / (2.0 * eps);
    // Build tangent frame at macro_n.  Standard branchless basis.
    float3 t1 = abs(macro_n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tx = normalize(cross(t1, macro_n));
    float3 tz = cross(macro_n, tx);
    // Bump along the tangent frame.  Empirically the noise gradient
    // magnitude is ~0.5..1.0, so multiplying by amp directly gives a
    // perceptual rate of perturbation matching the cvar's documented range.
    float3 perturbed = macro_n - amp * (dh_dx * tx + dh_dz * tz);
    return normalize(perturbed);
}

// --- Wave 8 ocean (#25) -- FFT heightfield ray-march -----------------------
// Bilinear, wrap-around fetch of the ocean textures in tile-UV space. The
// CPU FFT output tiles seamlessly, so we wrap with frac() and lerp four
// texels. Manual filtering (integer fetches + lerp) avoids needing a Slang
// SamplerState, matching how env_map is read by integer index elsewhere.
//
// The fetch is split into a target-agnostic weight computation
// (oceanBilinearTexels) + two thin per-texture wrappers, because the two
// ocean textures are declared as read-only Texture2D on Metal but
// RWTexture2D on Vulkan -- a single helper taking the texture as a
// parameter can't bind to both types across backends, so we index the
// global textures directly inside the per-texture wrappers.
struct OceanTexel {
    uint2 c00; uint2 c10; uint2 c01; uint2 c11;
    float tx;  float ty;
};
OceanTexel oceanBilinearTexels(float2 uv, uint grid) {
    OceanTexel o;
    float2 f = frac(uv);
    if (f.x < 0.0) f.x += 1.0;
    if (f.y < 0.0) f.y += 1.0;
    float gx = f.x * float(grid) - 0.5;
    float gy = f.y * float(grid) - 0.5;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    o.tx = gx - float(x0);
    o.ty = gy - float(y0);
    uint x0u = uint((x0 % int(grid) + int(grid)) % int(grid));
    uint y0u = uint((y0 % int(grid) + int(grid)) % int(grid));
    uint x1u = (x0u + 1u) % grid;
    uint y1u = (y0u + 1u) % grid;
    o.c00 = uint2(x0u, y0u);
    o.c10 = uint2(x1u, y0u);
    o.c01 = uint2(x0u, y1u);
    o.c11 = uint2(x1u, y1u);
    return o;
}
// --- Planetary P5 (#259): the cascade atlas --------------------------------
// With more than one cascade the two ocean textures become an ATLAS: grid
// wide, grid * cascade_count tall, cascade c occupying rows
// [c*grid, (c+1)*grid). Wrapping stays WITHIN the cascade's band, which is
// what oceanBilinearTexels' modulo already gives -- the row offset is added
// afterwards so a v-wrap can never bleed into the neighbouring cascade.
//
// A single stacked atlas rather than one texture pair per cascade because
// texture slots are the scarce resource here: PathTrace already saturates
// Apple Silicon's 8-read_write compute cap and pbr_atlas is deliberately
// declared last so it lands at MSL index 16 without moving anything. Three
// cascades as six textures would need four more slots and would move that
// assignment; as an atlas it needs none.
OceanTexel oceanBilinearTexelsC(float2 uv, uint grid, uint c) {
    OceanTexel o = oceanBilinearTexels(uv, grid);
    uint row = c * grid;
    o.c00.y += row; o.c10.y += row; o.c01.y += row; o.c11.y += row;
    return o;
}
// --- #259 / #133 Phase 2: the march bracket, out of the atlas -------------
//
// oceanRayMarchShell brackets the wave band at R_sea +/- h_max, where
// h_max is the SUM of the cascades' peak crests plus the host's slack.
// It used to arrive as ocean_params0.w, which worked only while the host
// solved the field: with shaders/OceanCascades.slang doing it the peak
// exists only after the reduce pass, and pushing last frame's number would
// put the bracket behind the field it brackets.
//
// So it travels in the displacement atlas, one row past the last cascade.
// That row is unreachable by any sample: oceanBilinearTexelsC wraps modulo
// `grid` INSIDE the cascade's band and only then adds c*grid, so the
// largest y any fetch can form is (count-1)*grid + grid-1. Both solvers
// write the identical float into the identical texel, so the shader reads
// one number from one place whoever produced the field.
float oceanShellHMax(uint count, uint grid) {
    return ocean_displacement[uint2(0u, count * grid)].x;
}
float4 oceanFetchDispC(float2 uv, uint grid, uint c) {
    OceanTexel o = oceanBilinearTexelsC(uv, grid, c);
    float4 c00 = ocean_displacement[o.c00];
    float4 c10 = ocean_displacement[o.c10];
    float4 c01 = ocean_displacement[o.c01];
    float4 c11 = ocean_displacement[o.c11];
    return lerp(lerp(c00, c10, o.tx), lerp(c01, c11, o.tx), o.ty);
}
float4 oceanFetchNormalC(float2 uv, uint grid, uint c) {
    OceanTexel o = oceanBilinearTexelsC(uv, grid, c);
    float4 c00 = ocean_normal[o.c00];
    float4 c10 = ocean_normal[o.c10];
    float4 c01 = ocean_normal[o.c01];
    float4 c11 = ocean_normal[o.c11];
    return lerp(lerp(c00, c10, o.tx), lerp(c01, c11, o.tx), o.ty);
}

// Cascade c's tile period in metres. Packed into the .w lanes of the three
// tangent-frame vectors rather than a fourth vec4 of its own.
float oceanCascadePeriod(uint c) {
    return (c == 0u) ? ocean_tan_e.w : ((c == 1u) ? ocean_tan_n.w
                                                  : ocean_tan_o.w);
}
// Cascade c's tile-space phase. Cascade 0's is exactly zero: the anchor
// lattice IS cascade 0's period, so the snap moves cascade 0 by a whole
// number of tiles and its lookup is invariant by construction.
float2 oceanCascadePhase(uint c) {
    return (c == 0u) ? float2(0.0, 0.0)
                     : ((c == 1u) ? ocean_phase.xy : ocean_phase.zw);
}

// --- Geometry-to-BRDF handover, Bruneton/Neyret/Holzschuch 2010 ------------
//
// How much of cascade c's slope variance a footprint of width `w` still
// RESOLVES, in [0, 1]. Not a smoothstep: it is the Phillips equilibrium
// range's own answer.
//
// In the equilibrium subrange the omnidirectional height spectrum goes as
// k^-4 (Phillips 1958, "The equilibrium range in the spectrum of
// wind-generated waves", J. Fluid Mech. 4(4) 426-434 -- and it is exactly
// the k^-4 that OceanFFT's Phillips P(k) carries). The slope variance
// integrand is then
//
//     k^2 * P(k) * k dk  ~  k^2 * k^-4 * k dk  =  dk / k  =  d(ln k),
//
// i.e. MEAN-SQUARE SLOPE IS UNIFORMLY DISTRIBUTED IN LOG WAVELENGTH. So
// the fraction of a band [lo, hi] that survives a footprint w is just the
// log-length of the part of the band above w:
//
//     g_c(w) = ln(hi / clamp(w, lo, hi)) / ln(hi / lo).
//
// One log per cascade, no tuned constant, and it is 1 at w <= lo and 0 at
// w >= hi exactly rather than asymptotically.
//
// The band edges are the partition the host built the cascades on: cascade
// c owns wavelengths from the NEXT finer cascade's period up to its own,
// and the finest cascade bottoms out at its own Nyquist (2 * grid spacing).
// kOceanBandDivisor MIRRORS Engine.h's constant of the same name: how many
// periods of its longest wave a cascade's tile holds. Cascade 0 is
// unbounded above, so its band top is its own tile period.
static const float kOceanBandDivisor = 8.0;
float oceanCascadeResolved(uint c, uint count, uint grid, float w) {
    float P  = oceanCascadePeriod(c);
    float hi = (c == 0u) ? P : (P / kOceanBandDivisor);
    float lo = (c + 1u < count)
                   ? (oceanCascadePeriod(c + 1u) / kOceanBandDivisor)
                   : (2.0 * P / float(max(grid, 1u)));
    if (!(hi > lo)) return 1.0;
    float wc = clamp(w, lo, hi);
    return log(hi / wc) / log(hi / lo);
}
// The three weights at once. Computed ONCE per ray, not per march step --
// the ray cone is constant over a hit.
float3 oceanCascadeResolvedAll(uint count, uint grid, float w) {
    float3 g = float3(0.0, 0.0, 0.0);
    if (count > 0u) g.x = oceanCascadeResolved(0u, count, grid, w);
    if (count > 1u) g.y = oceanCascadeResolved(1u, count, grid, w);
    if (count > 2u) g.z = oceanCascadeResolved(2u, count, grid, w);
    return g;
}

// The microfacet alpha^2 the water surface needs at this footprint.
//
// The whole handover is one conservation statement:
//
//     sigma^2_geometry(w) + alpha^2_brdf(w) = sigma^2_CoxMunk    for all w
//
// Geometry carries what the cascades still resolve; the BRDF carries the
// rest. Nothing fades, nothing is tuned, and the total is a MEASURED
// oceanographic constant (Cox & Munk 1954; see src/physics/OceanFFT.h for
// the alpha^2 = mean-square-slope derivation). At the boat the cascades
// carry almost all of it and the surface is a near-mirror; from orbit they
// carry none and alpha^2 is exactly Cox-Munk's sigma^2. There is no
// altitude anywhere in the expression -- only the cone.
float oceanBrdfAlpha2(float3 resolved) {
    float geom = dot(resolved, ocean_slope.xyz);
    return max(ocean_slope.w + planet_ocean.z - geom, 0.0);
}

float4 oceanFetchDisp(float2 uv, uint grid) {
    OceanTexel o = oceanBilinearTexels(uv, grid);
    float4 c00 = ocean_displacement[o.c00];
    float4 c10 = ocean_displacement[o.c10];
    float4 c01 = ocean_displacement[o.c01];
    float4 c11 = ocean_displacement[o.c11];
    return lerp(lerp(c00, c10, o.tx), lerp(c01, c11, o.tx), o.ty);
}
float4 oceanFetchNormal(float2 uv, uint grid) {
    OceanTexel o = oceanBilinearTexels(uv, grid);
    float4 c00 = ocean_normal[o.c00];
    float4 c10 = ocean_normal[o.c10];
    float4 c01 = ocean_normal[o.c01];
    float4 c11 = ocean_normal[o.c11];
    return lerp(lerp(c00, c10, o.tx), lerp(c01, c11, o.tx), o.ty);
}

// Sampled ocean surface height (world-space Y displacement, metres) at a
// world XZ point. plane_y is the analytic water-plane height; the displaced
// surface sits at plane_y + height.
float oceanHeightAt(float3 world_p, float plane_y) {
    float tile = ocean_params0.y;
    uint  grid = uint(max(ocean_params1.y, 1.0));
    float2 uv  = float2(world_p.x, world_p.z) / max(tile, 1e-3);
    float4 d   = oceanFetchDisp(uv, grid);
    return plane_y + d.y;
}

// --- Planetary P5 (#259): the crossing test, both ways --------------------
//
// `diff` is the signed height of the ray point above the displaced surface.
// The Wave 8 test was `diff_prev > 0.0 && diff_cur <= 0.0` -- above to below
// ONLY. A ray originating below the surface and travelling up starts with
// diff_prev < 0 and can never register a crossing, so it silently falls
// through to the analytic surface: Snell's window seen from underwater has
// been geometrically wrong since #25, before any planetary work.
//
// The fix is strictly additive. The first disjunct is the historic test
// character for character, so every ray that used to find a crossing finds
// the SAME crossing at the same step with the same secant weight; the second
// only fires where the first found nothing. diff_prev == 0 fires neither, as
// before.
bool oceanCrossed(float diff_prev, float diff_cur) {
    return (diff_prev > 0.0 && diff_cur <= 0.0) ||
           (diff_prev < 0.0 && diff_cur >= 0.0);
}
// Secant denominator, guarded away from zero on the side the sign of the
// crossing puts it. Above-to-below has diff_prev - diff_cur >= diff_prev > 0
// and takes max(den, 1e-5) -- the historic expression, unchanged. The new
// below-to-above case has den <= 0 and needs the guard mirrored, or the
// weight comes out negative-infinite instead of in [0, 1].
float oceanSecantDenom(float diff_prev, float diff_cur) {
    float den = diff_prev - diff_cur;
    return (den >= 0.0) ? max(den, 1e-5) : min(den, -1e-5);
}

// Result of the ocean heightfield ray-march.
struct OceanHit {
    bool   hit;
    float  t;          // ray parameter of the displaced-surface crossing
    float3 normal;     // FFT surface normal at the hit
    float  foam;       // foam coverage [0,1] at the hit
};

// --- Planetary P5 (#259): the camera-anchored tangent frame ---------------
//
// The FFT tile is intrinsically planar; the ocean is not. The parameterisation
// is an orthographic projection into the East/North basis at ONE anchor point
// on the shell, which the host recomputes each frame:
//
//     (e, n) = ( dot(p - O, East), dot(p - O, North) ),   metres.
//
// Distortion is second order in (d/R)^2 -- at d = 100 km that is 2.5e-4, i.e.
// 25 m of arc-length error over 100 km, on a field whose finest cascade tiles
// every 23 m and whose crests are sub-pixel past ~1.4 km anyway.
//
// WHY THE ANCHOR IS QUANTISED. A tangent origin that slides with the camera
// drags the wave field along with it and deletes parallax -- the sea would
// look painted on. The host snaps the anchor to a lattice whose spacing is
// exactly cascade 0's period, so the snap translates the field by a whole
// number of cascade-0 tiles and cascade 0's lookup does not move at all.
// Cascades 1 and 2 pick up a phase, which the host computes in f64 from the
// integer lattice indices and pushes -- a pure function of camera position,
// with no hysteresis, which is what the golden matrix requires.
//
// WHY THIS FIXES THE ABSOLUTE-XZ BUG TOO. The Wave 8 UV was
// `float2(world_p.x, world_p.z) / tile`; at an ECEF x of 6.4e6 over a 50 m
// tile that is 128 000, and frac() of it retains almost no sub-tile
// precision. Here the projection is taken against an anchor a few hundred km
// away at worst, and the integer tile offset travels separately in
// ocean_phase, so the mantissa is spent on the fractional part.
float2 oceanTangentEN(float3 p) {
    float3 d = p - ocean_tan_o.xyz;
    return float2(dot(d, ocean_tan_e.xyz), dot(d, ocean_tan_n.xyz));
}

// The summed cascade field at a point. `amp` is sqrt of each cascade's
// resolved-variance weight: scaling a cascade's amplitude by a scales its
// slope variance by a^2, so sqrt is what makes the variance bookkeeping in
// oceanBrdfAlpha2 exact rather than approximate.
struct OceanCascadeSample {
    float  height;     // summed vertical displacement, metres
    float2 slope;      // summed (dh/dEast, dh/dNorth)
    float  foam;
};
OceanCascadeSample oceanSampleCascades(float2 en, uint count, uint grid,
                                       float3 amp) {
    OceanCascadeSample s;
    s.height = 0.0; s.slope = float2(0.0, 0.0); s.foam = 0.0;
    [unroll]
    for (uint c = 0u; c < 3u; ++c) {
        if (c >= count) break;
        float a = (c == 0u) ? amp.x : ((c == 1u) ? amp.y : amp.z);
        if (a <= 0.0) continue;
        float2 uv = en / max(oceanCascadePeriod(c), 1e-3) + oceanCascadePhase(c);
        float4 d = oceanFetchDispC(uv, grid, c);
        float4 n = oceanFetchNormalC(uv, grid, c);
        s.height += a * d.y;
        // Recover the slope the packed unit normal encodes. The solver
        // builds it as normalize(-dh/dx, 1, -dh/dz), so the inverse is
        // exact. Slopes ADD across cascades; unit normals do not, which is
        // why the sum happens here and the normalise happens once at the
        // end.
        float ny = max(n.y, 1e-4);
        s.slope += a * float2(-n.x / ny, -n.z / ny);
        s.foam = max(s.foam, a * d.w);
    }
    return s;
}

// Height of the displaced sea surface above the shell, at a point, in
// metres. The band the march brackets is +/- ocean_params0.w about
// planet_ocean.x.
float oceanShellHeightAt(float3 p, uint count, uint grid, float3 amp) {
    return oceanSampleCascades(oceanTangentEN(p), count, grid, amp).height;
}

// World-space surface normal from the summed cascade slopes. `up`, `east`
// and `north` are the tangent frame; the slopes are expressed in it.
float3 oceanShellNormal(float2 slope, float3 up) {
    float3 n = up - slope.x * ocean_tan_e.xyz - slope.y * ocean_tan_n.xyz;
    return normalize(n);
}

// Ray-march the FFT ocean heightfield starting from (and bounded by) the
// analytic water-plane intersection. Standard relief-mapping linear search +
// one bisection refinement: step the ray downward through the displacement
// band [plane_y - h_max, plane_y + h_max], detecting the first sample where
// the ray crosses from above the surface to below it.
//
//   ro, rd   : ray origin / unit direction (world space).
//   plane_n  : analytic water-plane normal (used only to early-out on
//              edge-on rays).
//   plane_y  : world-space Y of the flat water plane.
//   plane_t  : ray parameter at the analytic plane crossing (far bound).
//   max_steps: linear-search step budget (ocean_params1.x).
OceanHit oceanRayMarch(float3 ro, float3 rd, float3 plane_n,
                       float plane_y, float plane_t, int max_steps) {
    OceanHit oh;
    oh.hit = false; oh.t = plane_t;
    oh.normal = plane_n; oh.foam = 0.0;

    float h_max = max(ocean_params0.w, 0.01);   // peak |height| + slack (m)
    uint  grid  = uint(max(ocean_params1.y, 1.0));
    float tile  = ocean_params0.y;

    // Rays nearly parallel to the surface: the heightfield march is
    // ill-conditioned (a tiny |rd.y| explodes the vertical search into a
    // huge t-range). Fall back to the analytic plane hit + FFT normal at
    // the plane crossing -- visually fine at the horizon where these rays
    // graze, and avoids a pathological step count.
    if (abs(rd.y) < 1e-3) {
        float3 pp = ro + rd * plane_t;
        float2 uv = float2(pp.x, pp.z) / max(tile, 1e-3);
        oh.normal = normalize(oceanFetchNormal(uv, grid).xyz);
        oh.foam   = oceanFetchDisp(uv, grid).w;
        oh.t = plane_t;
        oh.hit = true;
        return oh;
    }

    // Entry/exit of the displacement band along the ray. The surface lives
    // within plane_y +/- h_max; clamp the search to where the ray is inside
    // that slab so we don't waste steps in empty space.
    float t_top = (plane_y + h_max - ro.y) / rd.y;
    float t_bot = (plane_y - h_max - ro.y) / rd.y;
    float t_enter = min(t_top, t_bot);
    float t_exit  = max(t_top, t_bot);
    t_enter = max(t_enter, 0.0);
    // Never march past the analytic plane: it's the guaranteed backstop.
    t_exit  = min(t_exit, plane_t + h_max / max(abs(rd.y), 1e-3));
    if (t_exit <= t_enter) {
        // Degenerate band: just take the plane.
        oh.t = plane_t; oh.hit = true;
        float3 pp = ro + rd * plane_t;
        float2 uv = float2(pp.x, pp.z) / max(tile, 1e-3);
        oh.normal = normalize(oceanFetchNormal(uv, grid).xyz);
        oh.foam   = oceanFetchDisp(uv, grid).w;
        return oh;
    }

    int steps = clamp(max_steps, 8, 256);
    float dt  = (t_exit - t_enter) / float(steps);

    float t_prev = t_enter;
    float3 p_prev = ro + rd * t_prev;
    // diff > 0 means the ray point is ABOVE the surface.
    float diff_prev = p_prev.y - oceanHeightAt(p_prev, plane_y);

    bool found = false;
    float t_hit = plane_t;
    [loop]
    for (int i = 1; i <= steps; ++i) {
        float t_cur = t_enter + dt * float(i);
        float3 p_cur = ro + rd * t_cur;
        float diff_cur = p_cur.y - oceanHeightAt(p_cur, plane_y);
        if (oceanCrossed(diff_prev, diff_cur)) {
            // Crossing in [t_prev, t_cur]. One linear bisection on diff
            // gives a sub-step-accurate intersection (relief-mapping
            // secant step).
            float w = diff_prev / oceanSecantDenom(diff_prev, diff_cur);
            t_hit = lerp(t_prev, t_cur, w);
            found = true;
            break;
        }
        t_prev = t_cur;
        p_prev = p_cur;
        diff_prev = diff_cur;
    }

    if (!found) {
        // No crossing found within the band: this ray skims over the
        // wave crests and continues to the flat plane (or misses). Use
        // the analytic plane as the surface so water still reads as a
        // continuous sheet rather than dropping rays through to the sky.
        t_hit = plane_t;
    }

    float3 ph = ro + rd * t_hit;
    float2 uvh = float2(ph.x, ph.z) / max(tile, 1e-3);
    oh.normal = normalize(oceanFetchNormal(uvh, grid).xyz);
    oh.foam   = oceanFetchDisp(uvh, grid).w;
    oh.t      = t_hit;
    oh.hit    = true;
    return oh;
}

// --- Planetary P5 (#259): the radial march --------------------------------
//
// The same relief march, with "above the plane" replaced by "outside the
// shell". Three things change and one thing deliberately does not.
//
//  * The band is two SHELL intersections at R_sea +/- h_max rather than two
//    y-plane solves, via ptRayShell -- which carves the inner sphere out of
//    the outer one, so the returned interval is the near band crossing for a
//    ray coming down AND the upward band crossing for a submerged ray, with
//    no case analysis here.
//  * `p.y` becomes ptRayAltitudeAt: the cancellation-free altitude above the
//    shell, HOISTED along the ray. Materialising ro + rd*s and taking its
//    altitude would cap the march at ULP(|p|) ~ 0.5 m in any frame whose
//    origin is not near the surface -- against a 1.5 m wave.
//  * The crossing test is oceanCrossed, so a submerged ray looking up finds
//    the surface.
//
// AND THE GRAZING BAILOUT STAYS, and matters more here, not less: on a
// planet every horizon ray is grazing. Its criterion is now derived rather
// than an absolute 1e-3 on rd.y -- march the crests only while the band is
// wider than the pixel. 2*h_max <= cone_width is exactly the study's
// "displacement is sub-pixel beyond 1.4 km" bound written as the comparison
// it actually is (2 * 1.5 m / 1.07e-3 rad = 2.8 km at 60 deg / 1080p), and
// it degrades correctly at every resolution and field of view instead of at
// the one they were measured on. Past it the analytic shell IS the surface
// and only oceanBrdfAlpha2's roughness matters.
OceanHit oceanRayMarchShell(float3 ro, float3 rd, float3 shell_n,
                            float shell_t, float cone_w, int max_steps,
                            float3 resolved) {
    OceanHit oh;
    oh.hit = false; oh.t = shell_t;
    oh.normal = shell_n; oh.foam = 0.0;

    float R     = planet_ocean.x;
    float3 pc   = planet_center_radius.xyz;
    uint  grid  = uint(max(ocean_params1.y, 1.0));
    uint  count = uint(max(planet_ocean.y, 1.0));
    float h_max = max(oceanShellHMax(count, grid), 0.01);
    float3 amp  = sqrt(max(resolved, float3(0.0, 0.0, 0.0)));

    // Sub-pixel crests: take the shell, fetch the cascade normal there.
    // Note this is the FULL-amplitude normal at that point -- the roughness
    // fold has already removed the unresolved part from `amp`.
    if (2.0 * h_max <= cone_w) {
        float3 pp = ro + rd * shell_t;
        OceanCascadeSample s = oceanSampleCascades(oceanTangentEN(pp), count,
                                                   grid, amp);
        oh.normal = oceanShellNormal(s.slope, shell_n);
        oh.foam   = s.foam;
        oh.t      = shell_t;
        oh.hit    = true;
        return oh;
    }

    float t_in, t_out;
    if (!ptRayShell(ro, rd, pc, R - h_max, R + h_max, shell_t + h_max * 4.0,
                    t_in, t_out) || !(t_out > t_in)) {
        // Degenerate band (a ray that only clips the shell tangentially, or
        // one whose whole band lies behind it): take the analytic shell.
        float3 pp = ro + rd * shell_t;
        OceanCascadeSample s = oceanSampleCascades(oceanTangentEN(pp), count,
                                                   grid, amp);
        oh.normal = oceanShellNormal(s.slope, shell_n);
        oh.foam   = s.foam;
        oh.t      = shell_t;
        oh.hit    = true;
        return oh;
    }

    int steps = clamp(max_steps, 8, 256);
    float dt  = (t_out - t_in) / float(steps);

    // Hoisted altitude: |p(s) - c|^2 - R^2 is a quadratic in s whose two
    // coefficients are constant along the ray, so the accumulator is paid
    // once and each step costs two fma. See ptRayAltitudeBegin.
    PtRayAltitude alt = ptRayAltitudeBegin(ro, rd, pc, R);

    float t_prev = t_in;
    float3 p_prev = ro + rd * t_prev;
    // diff > 0 means the ray point is OUTSIDE the displaced surface.
    float diff_prev = ptRayAltitudeAt(alt, t_prev)
                    - oceanShellHeightAt(p_prev, count, grid, amp);

    bool found = false;
    float t_hit = shell_t;
    [loop]
    for (int i = 1; i <= steps; ++i) {
        float t_cur = t_in + dt * float(i);
        float3 p_cur = ro + rd * t_cur;
        float diff_cur = ptRayAltitudeAt(alt, t_cur)
                       - oceanShellHeightAt(p_cur, count, grid, amp);
        if (oceanCrossed(diff_prev, diff_cur)) {
            float w = diff_prev / oceanSecantDenom(diff_prev, diff_cur);
            t_hit = lerp(t_prev, t_cur, w);
            found = true;
            break;
        }
        t_prev = t_cur;
        diff_prev = diff_cur;
    }
    if (!found) t_hit = shell_t;

    float3 ph = ro + rd * t_hit;
    OceanCascadeSample s = oceanSampleCascades(oceanTangentEN(ph), count,
                                               grid, amp);
    // The macro normal is the local vertical AT THE HIT, not at the shell
    // entry: over a several-km band traverse those differ by the arc.
    oh.normal = oceanShellNormal(s.slope, localUp(ph));
    oh.foam   = s.foam;
    oh.t      = t_hit;
    oh.hit    = true;
    return oh;
}
#endif  // PT_WATER_ENABLED
// --- end Water Phase 1 -----------------------------------------------------

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
        // Motion blur (#85): the G-buffer pass writes the depth + motion
        // textures the denoiser uses for temporal reprojection. Pass
        // shutter_t01 = 1.0 (current frame position) so the depth read
        // by the denoiser stays stable from frame to frame -- a shutter
        // jittered into a half-streak position would feed the denoiser
        // a mismatched depth and the reprojection would smear. The
        // visible motion-blur streak is produced by the SHADING pass
        // (per-sample shutter_t01 inside the spp loop below).
        // Same camera cone as the shading pass below (#256): one pixel
        // of spread per unit distance, zero width at the eye.
        PtRayCone cone0 = ptConeMake(0.0,
                                     (2.0 * pos_fovtan.w) / float(max(dim.y, 1u)));
        HitInfo h0 = traceScene(ro0, rd0, 1.0, cone0);
        // Wave 8 ocean (#25): displace the G-buffer's water-plane hit too so
        // the denoiser's depth + normal match the shaded displaced surface
        // (otherwise the SVGF/MetalFX reprojection would see a flat plane
        // where the shading pass renders waves). Same gate + march as the
        // shading pass above.
#if PT_WATER_ENABLED
        // Planetary P5 (#259): the "is this surface horizontal enough for a
        // heightfield march" gate is dot(n, localUp) rather than n.y. In the
        // planar frame localUp() is float3(0,1,0) and the dot IS n.y bit for
        // bit, so every pre-P5 scene takes the identical branch.
        if (ocean_params0.x != 0.0 && h0.hit && h0.mat == MAT_WATER &&
            abs(dot(h0.normal, localUp(ro0 + rd0 * h0.t))) > 0.5) {
            if (planet_ocean.x > 0.0) {
                float3 res0 = oceanCascadeResolvedAll(
                    uint(max(planet_ocean.y, 1.0)),
                    uint(max(ocean_params1.y, 1.0)),
                    ptConeWidthAt(cone0, h0.t));
                OceanHit oo0 = oceanRayMarchShell(ro0, rd0, h0.normal, h0.t,
                                                  ptConeWidthAt(cone0, h0.t),
                                                  int(ocean_params1.x), res0);
                if (oo0.hit) { h0.t = oo0.t; h0.normal = oo0.normal; }
            } else {
                float plane_y0 = (ro0 + rd0 * h0.t).y;
                OceanHit oo0 = oceanRayMarch(ro0, rd0, h0.normal, plane_y0, h0.t,
                                             int(ocean_params1.x));
                if (oo0.hit) { h0.t = oo0.t; h0.normal = oo0.normal; }
            }
        }
#endif

        float3 hit_normal;
        float3 hit_albedo;   // linear RGB at primary hit; sky writes float3(0)
        float  depth_view;   // linear camera-space Z (world units along forward)
        // Homogeneous world position fed to the reprojection below. A
        // surface hit is a finite point (w = 1); the sky is a point at
        // INFINITY along the primary ray, which in homogeneous
        // coordinates is exactly float4(rd0, 0) -- see the else branch.
        float4 reproj_h;
        if (h0.hit) {
            float3 hit_world = ro0 + rd0 * h0.t;
            reproj_h   = float4(hit_world, 1.0);
            hit_normal = h0.normal;
            hit_albedo = h0.albedo;
            depth_view = h0.t * dot(rd0, normalize(fwd_aspect.xyz));
        } else {
            // Sky / stars motion vector (#254). This used to reproject a
            // proxy point 1 km down the primary ray, which manufactures
            // parallax the sky does not have: the sky is at infinity, so
            // its only apparent motion is camera ROTATION. Feeding a
            // finite proxy through a translating camera makes SVGF's
            // temporal pass and MetalFX's upsampler reproject every sky
            // pixel to the wrong place, and they then either ghost or
            // reject history wholesale. At a 400 km circular orbit
            // (v = 7 672 m/s, mu = 3.986004418e14 m^3/s^2, r = 6.771e6 m)
            // the camera translates 128 m per 60 fps frame, which against
            // a 1 km proxy is ~7 deg of false parallax EVERY frame; a
            // fast ground camera hits a milder version of the same bug.
            //
            // The fix is exact and free: reproject the direction with
            // w = 0. Both view-projection matrices then contribute only
            // their rotational part (the translation column multiplies
            // w), so the motion vector carries rotation-only parallax.
            // The CURRENT-frame pixel position is unchanged by this --
            // the camera origin maps to the view-space origin, so a point
            // at t along the ray and the ray direction itself project to
            // the same NDC xy for every t > 0; only the homogeneous
            // scale differs, and that divides out.
            //
            // The depth sentinel below was already raised from 1000 to
            // 1e10 for the related reason; the motion proxy was simply
            // left behind.
            reproj_h   = float4(rd0, 0.0);
            hit_normal = float3(0.0, 0.0, 0.0); // sentinel: sky / no surface
            hit_albedo = float3(0.0, 0.0, 0.0); // sky has no surface albedo
            // Sky sentinel: a value impossibly large for real geometry so
            // downstream consumers (StarsComposite's sky-pixel gate,
            // SVGF's edge-stops) can unambiguously distinguish "ray
            // missed" from "ray hit something very far away." The
            // previous 1000.0 sentinel collided with real depths: a
            // camera at altitude looking toward the horizon sees ground
            // at hundreds of meters of camera-space Z (a grazing ray
            // skims the ground for a long distance before hitting), and
            // anything > 1000 m of legitimate geometry tripped the
            // composite's gate. 1e10 m is comfortably outside any
            // plausible engine scene scale (1 unit = 1 metre) and small
            // enough that downstream numeric ops (depth deltas,
            // edge-stop exponentials) stay in safe float32 range.
            //
            // This comment used to justify 1e10 by claiming it was past
            // the outer planets (issue #276). It is not: 1e10 m is
            // 0.067 AU, INSIDE Mercury's orbit (0.39 AU), and roughly
            // 590x short of the 39 AU that claim named. The sentinel
            // value itself is fine for its purpose -- the yardstick that
            // justified it was not, and the true ceiling is worth
            // recording because it is a real one: geometry at 1.5e11 m
            // (1 AU) would collide with this sentinel, a landmine for
            // anyone later adding a body you can fly to.
            depth_view = 1.0e10;
        }

        // Pixel-space motion via reprojection. We still need the
        // perspective matrices for this even though depth itself goes
        // to MetalFX in linear units.
        float4 pc = mul(curr_view_proj, reproj_h);
        float4 pp = mul(prev_view_proj, reproj_h);
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
        // Albedo G-buffer: allocated/bound by the engine for OptiX AOV
        // variants, MetalFX (and SVGF->MetalFX chained kinds), AND --
        // after issue #119 -- the pure SVGF/NRD kinds, which use the
        // albedo for demodulation (divide-by-albedo into the denoiser
        // chain, multiply back on output). The runtime
        // `write_albedo_gbuffer` push gate is the actual run-or-skip
        // switch -- the engine sets it to 1 whenever albedo_tex_id_
        // is non-zero (regardless of backend). Vulkan tolerates the
        // unbound slot for non-consumer kinds via
        // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT; Metal binds a
        // placeholder when albedo_tex_id_ is 0.
        //
        // CRITICAL: keep this write code visible to ALL targets, NOT
        // wrapped in `#ifdef PT_TARGET_SPIRV`. When the write was
        // SPIR-V-only, Slang's MSL backend saw albedo_tex as a
        // declared-but-never-used resource and dead-code-eliminated
        // it from the kernel signature -- which compacted all
        // following texture slots by one and silently broke every
        // engine-side BindStorageTexture call that landed past it.
        // Bug history: this caused PR #108's star-split slot 10
        // binding to land on the wrong MSL texture, rendering zero
        // stars on Mac under any denoiser. The fix is to keep the
        // write site reachable on both backends (write is still
        // runtime-gated so non-AOV dispatches don't pay for it).
        // Removing the gate ENTIRELY would also work but costs an
        // RGBA16F write per pixel per frame on non-AOV
        // modes; the runtime gate keeps that free.
        if (write_albedo_gbuffer != 0u) {
            albedo_tex[tid] = float4(hit_albedo, 0.0);
        }
        // MetalFX specular-guidance G-buffers (issue #118). Three
        // textures fed to MTLFXTemporalDenoisedScaler so it can tell
        // specular from diffuse response -- without them the scaler
        // produces 8x8 halos around bright reflections / metals.
        //
        // F0 (specular_albedo) per material kind:
        //   MAT_METAL      : F0 = albedo (the material colour reflects
        //                    directly; gold's yellow specular IS its
        //                    yellow albedo, not a tinted white highlight).
        //   MAT_DIELECTRIC : F0 = float3(0.04) — the standard Schlick
        //                    approximation for IOR~1.5 (glass, plastic,
        //                    coated dielectrics). Engine's dielectric
        //                    primitives are nominally IOR=1.5 unless
        //                    overridden in the prim material struct.
        //   MAT_LAMBERT    : F0 = float3(0) — no specular lobe at all.
        // Pulled from h0 (not hit_albedo / hit_normal which collapse
        // to sentinels on sky pixels) so we can preserve the material-
        // kind discriminator.
        //
        // roughness: from h0.roughness directly. The material struct
        // packs roughness in primitive .v1.a / SDF cluster header /
        // mesh material — already plumbed through traceScene as h.roughness.
        //
        // specular_hit_distance: MVP proxy using `primary_t * (1 - rough)`,
        // i.e. attenuated by smoothness so a perfect mirror keeps the
        // full distance and a rough surface trends toward 0 where
        // MetalFX has no specular reproject to do. A future PR (real
        // reflection-ray trace from the primary hit) can replace this
        // with the actual reflected-virtual-image depth; see the
        // texture's declaration comment for the design trade-off.
        //
        // All three are sky-aware: hit_normal == float3(0) means h0.hit
        // == false, so we write the "no surface" sentinel for that
        // pixel. roughness sentinel is 1.0 (fully rough -> MetalFX
        // skips specular handling); F0 + hit_distance sentinel is 0.
        if (write_specular_albedo_gbuffer != 0u) {
            float3 f0 = float3(0.0);
            if (h0.hit) {
                if      (h0.mat == MAT_METAL)      f0 = h0.albedo;
                else if (h0.mat == MAT_DIELECTRIC) f0 = float3(0.04);
                // Lambert keeps f0 = 0.
            }
            // WTexture2D.Store(coord, value) instead of subscript
            // assignment -- WTexture2D is write-only (lowers to Metal
            // access::write to stay under the 8-RW-texture cap) and
            // doesn't support the [] = ... HLSL convenience syntax.
            specular_albedo_tex.Store(tid, float4(f0, 0.0));
        }
        if (write_roughness_gbuffer != 0u) {
            float r = h0.hit ? saturate(h0.roughness) : 1.0;
            // Lambert lobe is fully rough; the engine's prim seeding
            // sets h.roughness based on the material's GGX param, but
            // for MAT_LAMBERT the value is meaningless (Lambert doesn't
            // have a roughness). Force = 1.0 so MetalFX sees Lambert
            // pixels as "no specular surface to preserve."
            if (h0.hit && h0.mat == MAT_LAMBERT) r = 1.0;
            roughness_tex.Store(tid, r);
        }
        if (write_specular_hit_distance_gbuffer != 0u) {
            float dist = 0.0;
            if (h0.hit) {
                // MVP proxy: primary hit distance weighted by smoothness
                // (1 - roughness). At roughness=0 the reflection comes
                // from somewhere far past the primary hit, but we don't
                // trace it -- the value is "as far as we know, primary_t."
                // At roughness=1 we'd write 0 anyway since MetalFX won't
                // try to reproject a fully-rough specular.
                float smoothness = 1.0 - saturate(h0.roughness);
                dist = h0.t * smoothness;
            }
            specular_hit_distance_tex.Store(tid, dist);
        }
    }

    // Multi-sample loop: average `spp` independent paths into a single
    // radiance estimate. spp>1 buys cleaner frames during motion at a
    // proportional GPU cost; spp=1 is the classic 1-spp interactive mode.
    uint sample_count = (spp == 0u) ? 1u : spp;
    float3 frame_radiance = float3(0.0);
    // SIGMA sun-shadow demodulation accumulators (issue #115, B10 fix).
    // Only touched when write_shadow_vis != 0 (Metal SIGMA path, procedural
    // sun). frame_sun_direct is the UNSHADOWED primary-hit direct-sun
    // radiance S; it is packed into shadow_vis_buf and re-modulated
    // post-denoise by SigmaShadow.slang (hdr += S * vis_denoised).
    // frame_sun_shadowed is the SHADOWED direct-sun contribution actually
    // folded into frame_radiance at the primary hit; it is SUBTRACTED from
    // the denoiser input so SVGF never sees -- and so never spatially
    // bleeds -- the hard sun-shadow edge. frame_radiance itself (and thus
    // accum_hdr / auto-exposure / the denoiser-off display path) is left
    // bit-identical to the pre-fix code. See the demodulation write below.
    float3 frame_sun_direct   = float3(0.0);
    float3 frame_sun_shadowed = float3(0.0);
    // Celestials composite gate (issue #46). When set, the b==0
    // primary-miss code below subtracts starsOnly + sunDisc + moonDisc
    // from the sky term so denoise_color reaches the denoiser without
    // sub-pixel celestials. The engine's StarsComposite kernel
    // (dispatched post-Denoise, pre-bloom on the Metal use_engine_tonemap
    // path) re-adds them with N aperture-sampled rays per pixel.
    // Gated on denoiser_enabled + write_hdr_aux because the legacy
    // (denoiser-off) path's inline tonemap already produces sharp
    // celestials at 1 spp; the smudge is only a problem under SVGF/
    // MetalFX.
    const bool composite_celestials_active =
        (denoiser_enabled != 0u) && (write_hdr_aux != 0u) &&
        (composite_celestials != 0u);
    // Cloud transmittance G-buffer (issue #46 follow-up). Captures the
    // camera-ray cloud march's running trans_eye_c product so the post-
    // denoise StarsComposite kernel can attenuate the celestial
    // composite by it -- without this, stars / sun / moon punch through
    // clouds at full brightness in the final image. Defaults to 1.0
    // (no occlusion) when clouds are off, the ray misses the cloud
    // layer, or no march runs. Captured from sample 0 only -- the
    // cloud field is smooth enough that the sub-pixel jitter delta
    // between samples is well below cloud feature size, so any single
    // sample is a faithful proxy for the unjittered camera ray.
    float frame_cloud_trans = 1.0;
    // Sun-NEE shadow-ray visibility accumulator (issue #115). Sums the
    // primary-hit shadow-ray transmittance toward the sun across all
    // spp samples; divided by sample_count at the end of main() to give
    // a per-pixel [0,1] visibility for the SigmaShadow denoiser to
    // bilateral-filter. Per-sample contributions:
    //   - sky-miss primary ray  : 1.0 (no surface = no shadow to denoise)
    //   - sun below horizon     : 1.0 (no procedural sun visibility)
    //   - surface hit + sun up  : scalar luminance of the sun-NEE
    //                             shadow-ray transmittance, traced for
    //                             ANY primary hit regardless of material.
    //
    // Material-agnostic by design: metal and dielectric primary hits
    // also contribute the real visibility (not 1.0). Lambert-only
    // gating was the original intent and would have left specular /
    // glass surfaces undenoised, with sharp direct-shadow streaks
    // visible against smooth indirect lighting. Tracing the visibility
    // ray for every h.hit case captures the same direct-light signal
    // the SigmaShadow filter is designed to bilateral-smooth, and the
    // demod compose later multiplies HDR by this scalar regardless of
    // BSDF -- so the post-denoise HDR carries the same shadow boundary
    // sharpness on every material. Cost is one extra transmittance()
    // call per primary hit, but only when write_shadow_vis != 0 (engine
    // gates that on procedural sky + denoiser + r_shadow_demod).
    //
    // Initialized to 0.0; sky pixels accumulate sample_count*1.0, then
    // divide back to 1.0 below. Future material gating (e.g. mirrored
    // metal should arguably bypass the demod) is documented as a
    // follow-up rather than implemented here, to keep this PR's scope
    // tight to Lambert-and-friends.
    float frame_shadow_vis = 0.0;
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
        // --- Wave 8 spectral (#27) ---
        // Per-ray hero-wavelength sampling. Each path samples ONE wavelength
        // lambda in [380, 730] nm, stratified across the pixel's spp samples
        // for fast spectral convergence. The dielectric BSDF (below) bends
        // that wavelength by its own Cauchy IOR -> dispersion; the path's
        // RGB radiance is then weighted by the CIE-derived per-wavelength
        // sRGB response (spectral_weight) so averaging many wavelengths
        // reconstructs full colour. spectral_weight is normalised so a flat
        // (non-dispersive) spectrum reconstructs the achromatic RGB image in
        // expectation -- which is why spectral mode only diverges from the
        // RGB path where IOR actually varies with lambda.
        //
        // Strict bit-stability when OFF: the heroWavelength() call consumes
        // one randf(seed) draw, so it is gated entirely behind the host flag.
        // With spectral off, hero_lambda stays at the sentinel, the dielectric
        // path uses h.ior unchanged, spectral_weight is (1,1,1), and NO extra
        // seed draw happens -- the legacy RGB path is preserved bit-for-bit.
        float  hero_lambda    = -1.0;          // <0 sentinel = spectral off
        float3 spectral_weight = float3(1.0);
        if (spectral_params.x > 0.5) {
            hero_lambda     = heroWavelength(s, sample_count, seed);
            spectral_weight = spectralWeightSRGB(hero_lambda);
        }
        // --- end ---
        // Motion blur (#85): pick a uniform shutter-time sample per ray.
        // 0 = beginning of the shutter window, 1 = end. When the host
        // gate is off (motion_blur_params.x == 0) we pin t01 to 1.0 so
        // the sphere center lerp collapses to v0.xyz (current frame
        // position), which is the bit-exact pre-#85 sampling. The
        // shutter_speed cvar (motion_blur_params.y) is intentionally
        // not folded into the lerp factor: the prev/curr snapshot
        // already encodes the per-frame displacement, and a longer
        // shutter just means the user is happy with the streak length
        // produced by that displacement at the current frame rate.
        float shutter_t01 = (motion_blur_params.x > 0.5)
                                ? randf(seed)
                                : 1.0;

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
            // --- Wave 10 bloom/bokeh ---
            // Round aperture (blades < 3): concentric-disk mapping
            // (Shirley-Chiu) of the square (u1,u2) onto a uniformly
            // distributed unit disk -- bit-identical to the pre-wave-10
            // sampler. Polygonal iris (blades >= 3): an AREA-UNIFORM sample
            // of the regular N-gon drawn directly from (u1,u2) via the
            // wedge/triangle method (rotated by the blade orientation
            // dof_params.w), giving the classic 5-/6-/8-sided bokeh with a
            // physically-correct (unbiased) uniform density. Both branches
            // consume exactly the same two uniforms, so the seed stream
            // after this block is identical regardless of blade count.
            // (The earlier disk->polygon radial warp was area-biased -- the
            // map's Jacobian varied with the sample angle, over-weighting the
            // polygon toward its vertices and skewing DOF integration. See
            // PR #235.)
            float2 ap = (blades >= 3)
                          ? ptPolygonSample(u1, u2, blades, dof_params.w)
                          : ptConcentricDisk(u1, u2);
            // Cat's-eye / optical vignetting: off-axis image points see a
            // pupil partly occluded by the rear elements, clipping the
            // bokeh into a lemon shape squeezed toward frame centre. uv is
            // the pixel's normalised [-1,1] screen position, so |uv| is the
            // image eccentricity. Strength 0 leaves the bokeh round
            // (bit-for-bit with the pre-wave-10 aperture).
            if (dof_bokeh_params.x > 0.0) {
                ap = ptCatEye(ap, uv, dof_bokeh_params.x);
            }
            // --- end Wave 10 bloom/bokeh ---
            float3 lens_off = right_xyz.xyz * (ap.x * aperture)
                            + up_xyz.xyz    * (ap.y * aperture);
            float3 focus_p  = ro + rd * focal_d;
            ro = ro + lens_off;
            rd = normalize(focus_p - ro);
        }

        float3 throughput = float3(1.0);
        float3 radiance   = float3(0.0);
        // SIGMA demod per-sample captures (issue #115, B10). Populated at
        // the primary hit (b==0) inside the Lambert sun-NEE block, then
        // carried through the same multiplicative post-loop scalings
        // `radiance` gets (trans_atm_surf, trans_eye_c) so they stay
        // consistent with the sun term inside `radiance`. Zero for
        // non-Lambert primary hits, sky pixels, sun below horizon, and
        // whenever write_shadow_vis == 0.
        float3 sample_sun_direct   = float3(0.0);   // unshadowed sun S
        float3 sample_sun_shadowed = float3(0.0);   // shadowed sun in radiance
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
        // Planetary P5 (#259): the previous bounce was a planetary-ocean
        // specular hit that integrated the sun by NEE, so the sun DISC must
        // come out of this segment's sky or it is counted twice. Exactly one
        // segment of scope -- snapshotted and cleared at the top of the loop
        // body -- because the suppression belongs to the ray the water
        // launched and to nothing after it.
        bool   ocean_sun_nee = false;
        // --- Underwater medium tracking (Phase 4, #134; P7 #261) -------------
        // Per-ray "which medium is this segment travelling through?" stack.
        // Initialised from water_state.x (host scans analytic primitives at
        // push setup time and sets 1.0 when the camera position is below
        // any MAT_WATER plane, or inside the planetary sea-level shell).
        // Pushed / popped by the MAT_WATER and MAT_DIELECTRIC refract paths
        // in the bounce loop below.
        //
        // When the stack's top is water the per-bounce step:
        //   * applies exp(-water_params0.xyz * segment_length) to throughput
        //     over the segment just traversed (replaces the old nominal
        //     1.5m entry-side estimate with the exact distance the ray
        //     travelled before hitting the next surface)
        //   * sky-on-miss returns water_state.yzw (deep-water tint) instead
        //     of skyColor(ro, rd, seed), so TIR rays from underwater do not
        //     escape to
        //     the bright horizon when the scene runs out of geometry below
        //     the water plane.
        //
        // TIR at MAT_WATER leaves the stack alone (the ray stays in the same
        // medium); only the refract branches move it.
        //
        // Planetary P7 (#261) REPLACES the Phase 4 one-bit `in_water`. That
        // bit's own docstring called MAT_DIELECTRIC's exclusion "a Phase 4
        // simplification" and said a proper stack would land later; this is
        // later. The bit could not express "in water, inside a bubble", so a
        // submerged glass body kept attenuating with water absorption over
        // the segment INSIDE the glass, and every interface assumed the far
        // side was air. See ptMediumPush / ptMediumPop near MAT_WATER.
        uint medium = (water_state.x > 0.5)
                          ? ptMediumPush(0u, MEDIUM_WATER)
                          : 0u;
        // Whether the camera primary ray was underwater at ray-gen.
        // Latched once so the post-bounce atmospheric / volumetric march
        // can skip the surface aerial perspective + Mie/Rayleigh in-
        // scatter terms when the primary segment was traversed through
        // water (already attenuated by per-segment Beer's law inside the
        // bounce loop; atmospheric terms would double-dim and add
        // air-only Mie/Rayleigh in-scatter to a water-only segment).
        //
        // Pinned to false when PT_WATER_ENABLED is compiled off: the
        // water BRDF + in-loop segment absorption + sky-miss override
        // are all excised at compile time, so even if the host still
        // sets water_state.x from a MAT_WATER plane scan, this build
        // has no water medium to actually attenuate the primary ray.
        // Skipping the atmospheric march without compensating would
        // strip aerial perspective / volumetric in-scatter from
        // unrelated scenes that happen to mention `water` in a cfg.
#if PT_WATER_ENABLED
        bool primary_in_water = ptMediumIsWater(medium);
        // The stack AS AT RAY-GEN. `medium` is mutated by every refraction
        // in the bounce loop below, and the post-loop in-scatter march
        // needs the medium the PRIMARY segment ran through, not whatever
        // the last bounce left behind.
        uint primary_medium = medium;
#else
        bool primary_in_water = false;
        uint primary_medium = 0u;
#endif
        // --- end Underwater medium tracking ---------------------------------

#if PT_WATER_ENABLED
        // --- #305: water-leaving radiance, for a camera OUTSIDE the medium --
        //
        // THE HOLE THIS FILLS. Every in-scatter term above is gated on
        // primary_in_water, so a ray that refracts in from OUTSIDE got
        // Beer's law and nothing else: absorption with no source. Ocean
        // colour is not a tint on the transmitted background, it IS the
        // source term -- light enters, red is gone within metres (Pope &
        // Fry 1997), blue scatters back out (Morel 1974). With no source
        // the sea from orbit is whatever the sky reflects off it, which
        // is why #305 measured the water body contributing NO blue at all.
        //
        // WHY THREE FLOATS AND NOT A MARCH. The submerged march below
        // spends N shadow rays per pixel, which is affordable when the
        // camera is in the water and the water is a few pixels of the
        // frame -- and is not affordable when the ocean IS the frame, as
        // it is from orbit. For a HOMOGENEOUS column the single-scatter
        // integral the march evaluates numerically has a closed form, so
        // this path evaluates it exactly instead, in O(1) plus at most one
        // shadow ray. Derivation and the error bound against the march are
        // in src/physics/WaterOptics.h section 5; the identity is asserted
        // to 8 significant figures in tests/pt_water_optics_test.cpp.
        //
        //   wl_amp  = the radiance that leaves an INFINITELY deep column,
        //             already divided by n^2 (the radiance-scaling factor
        //             for water -> air) but NOT multiplied by the upward
        //             Fresnel transmittance -- see below for why.
        //   wl_rate = the per-metre-of-PATH rate at which the depth
        //             integral saturates, so a finite column of path
        //             length T contributes wl_amp * (1 - exp(-wl_rate T)).
        //             This is what makes a swimming pool shallow and the
        //             open ocean deep with one expression and no regime
        //             switch.
        //
        // WHY THE FRESNEL FACTOR IS ABSENT FROM wl_amp. The MAT_WATER
        // interface already picks reflect-vs-refract by Russian roulette
        // on the Fresnel reflectance with NO compensating weight, which is
        // the unbiased estimator for the (F, 1-F) pair. Adding this term
        // on the REFRACT branch therefore already carries its (1 - F)
        // factor as a branch probability; writing (1 - F) into wl_amp as
        // well would square it. Fresnel is reciprocal, so the F the split
        // used for the downward crossing is the same F the upwelling
        // radiance meets on the way out -- one number, used once.
        //
        // TOTAL INTERNAL REFLECTION cannot bite here and that is a
        // property of the geometry, not an omission: the in-water
        // direction is obtained by refracting a real air-side view ray, so
        // it is inside Snell's window by construction (asin(1/1.34) =
        // 48.3 deg) and always has a transmitted lobe. What IS dropped is
        // the light that TIRs back DOWN off the underside of the surface
        // and scatters out on a later attempt -- the multiple-internal-
        // reflection factor 1/(1 - r_bar * R) of the classical treatment.
        // At the open ocean's irradiance reflectance that factor is under
        // 1.02; it is bounded in WaterOptics.h rather than waved at.
        bool   wl_pending = false;
        bool   wl_primary = false;
        float3 wl_amp     = float3(0.0);
        float3 wl_rate    = float3(0.0);
#endif

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

        // Ray-cone texture LOD state (RT Gems 2019, ch. 20). cone_spread is
        // the angular size of one pixel (isometric for square pixels:
        // 2*tan(fovY/2)/height); cone_width is the cone diameter at the
        // current ray origin, grown by spread*segment each hit so deeper
        // bounces read progressively coarser mips. Reset per camera sample.
        float cone_spread = (2.0 * pos_fovtan.w) / float(max(dim.y, 1u));
        float cone_width  = 0.0;

        for (uint b = 0u; b < max_bounces; ++b) {
        // Issue #256: the same cone that drives texture LOD is what sets
        // every ray epsilon on this segment. cone_width is the footprint
        // at THIS ray's origin (0 at the eye, grown by the accumulate a
        // few lines below each time the path bounces).
        PtRayCone cone = ptConeMake(cone_width, cone_spread);
        // Planetary P5 (#259): consume the previous segment's sun-disc
        // suppression and clear it in the same breath, so it can never
        // outlive the ray it belongs to.
        bool ocean_sun_nee_prev = ocean_sun_nee;
        ocean_sun_nee = false;
        HitInfo h = traceScene(ro, rd, shutter_t01, cone);
        // --- Wave 8 ocean (#25) -- displace the analytic water plane --------
        // When the FFT ocean is on (ocean_params0.x != 0) and the ray hit a
        // MAT_WATER analytic plane, ray-march the displaced FFT heightfield
        // off that plane and overwrite h.t / h.normal with the displaced
        // crossing + FFT surface normal. Doing this BEFORE primary_t is
        // recorded (and before hit_pt / the MAT_WATER BRDF branch below)
        // flows the displaced depth + normal through the G-buffer, aerial
        // perspective, Beer's-law segment, and shading uniformly. h.albedo's
        // alpha carries the per-hit foam coverage for the BRDF branch.
        float ocean_foam_here = 0.0;
        // Planetary P5 (#259): the microfacet alpha^2 the sub-footprint wave
        // slope contributes at THIS hit. Zero everywhere except a planetary
        // ocean hit, and the water BRDF's rough branch is gated on it being
        // positive, so every pre-P5 water surface keeps its mirror path.
        float ocean_alpha2_here = 0.0;
#if PT_WATER_ENABLED
        if (planet_ocean.x > 0.0 && h.hit && h.mat == MAT_WATER) {
            // Planetary P5 (#259). The roughness fold is NOT gated on the
            // FFT being on, and that is what makes the acceptance test
            // possible rather than merely plausible: with r_ocean 0 the
            // cascades resolve nothing, so alpha^2 is Cox & Munk's sigma^2
            // at every distance and the shell is the smooth glitter sphere a
            // wave field seen from far enough away IS. Turning r_ocean on
            // must not change what the frame looks like from 400 km -- the
            // two halves of the same conservation law.
            float ocean_cone_w = ptConeWidthAt(cone, h.t);
            float3 res = float3(0.0, 0.0, 0.0);
            if (ocean_params0.x != 0.0) {
                res = oceanCascadeResolvedAll(uint(max(planet_ocean.y, 1.0)),
                                              uint(max(ocean_params1.y, 1.0)),
                                              ocean_cone_w);
            }
            ocean_alpha2_here = oceanBrdfAlpha2(res);
            if (ocean_params0.x != 0.0 &&
                abs(dot(h.normal, localUp(ro + rd * h.t))) > 0.5) {
                OceanHit oo = oceanRayMarchShell(ro, rd, h.normal, h.t,
                                                 ocean_cone_w,
                                                 int(ocean_params1.x), res);
                if (oo.hit) {
                    h.t = oo.t;
                    h.normal = oo.normal;
                    ocean_foam_here = oo.foam;
                }
            }
        } else if (ocean_params0.x != 0.0 && h.hit && h.mat == MAT_WATER &&
                   abs(dot(h.normal, localUp(ro + rd * h.t))) > 0.5) {
            float plane_y = (ro + rd * h.t).y;   // analytic plane height
            OceanHit oo = oceanRayMarch(ro, rd, h.normal, plane_y, h.t,
                                        int(ocean_params1.x));
            if (oo.hit) {
                h.t = oo.t;
                h.normal = oo.normal;
                ocean_foam_here = oo.foam;
            }
        }
#endif
        // --- end Wave 8 ocean -----------------------------------------------
        if (b == 0u) {
            // Sky pixels (no hit) keep the 50km default so cloud
            // marches reach the layer; object hits override.
            primary_t = h.hit ? h.t : primary_t;
            primary_hit = h.hit;
            primary_t_recorded = true;
            // Sun-NEE shadow visibility (issue #115). One dedicated
            // shadow ray per spp sample from the primary hit toward the
            // jittered sun-disc direction. This is INDEPENDENT of the
            // Lambert NEE branch's shadow ray below -- we want a clean
            // visibility signal at the primary surface regardless of
            // material so the SigmaShadow denoiser can produce sharp
            // sun shadows on metal and dielectric surfaces too (which
            // a Lambert-only NEE-tap wouldn't capture). Cost is one
            // extra `transmittance()` call per primary hit, BUT only
            // when r_shadow_demod is on -- gating on write_shadow_vis
            // keeps the legacy (denoiser-off or demod-off) path
            // bitwise-identical to pre-#115 by not touching `seed` or
            // executing any extra GPU work.
            //
            // Contribution rules:
            //   - sky pixel (no primary hit): 1.0 (no occlusion to denoise)
            //   - sun below horizon: 1.0 (no shadow possible)
            //   - surface hit + sun up: scalar luminance of the
            //     transmittance vector (RGB tint from glass etc.
            //     reduces to a single visibility scalar -- the demod
            //     compose is a per-channel-equal multiply by design,
            //     full chromatic shadows are a follow-up).
            //
            // Origin is offset by ptRayOrigin (#256) -- the same
            // scale-relative offset the NEE branch below uses, not the
            // absolute millimetre this used to carry. cone_width has not
            // been accumulated for this segment yet at this point in the
            // loop, so the footprint at the hit is spread * h.t. For the
            // b==0 hit we don't yet have the "flipped" normal nf, so
            // recompute it inline.
            if (write_shadow_vis != 0u) {
                if (h.hit) {
                    float3 sun_dir = sun_and_mode.xyz;
                    float3 hp      = ro + rd * h.t;
                    // Planetary P0 (#254): "is the sun up" is asked at the
                    // SHADED POINT, not of the frame. Planar frame ->
                    // sun_dir.y bit-for-bit; `hp` was already computed on
                    // this branch, only its scope moved.
                    if (elevationAt(hp, sun_dir) > 0.0) {
                        float ndi_0 = dot(h.normal, rd);
                        float3 nf_0 = (ndi_0 < 0.0) ? h.normal : -h.normal;
                        // sampleSunDisc jitters within the sun's
                        // angular footprint so the visibility
                        // integrates to a soft penumbra when the
                        // surface is near the sun's edge shadow.
                        // Matches what the NEE branch later does so
                        // both signals converge to the same noiseless
                        // mean.
                        float3 sun_jit = sampleSunDisc(sun_dir, seed);
                        PtRayCone cone_h = ptConeMake(cone_spread * h.t,
                                                      cone_spread);
                        float3 t_vis = transmittance(
                            ptRayOrigin(hp, nf_0, cone_h, sun_jit),
                            sun_jit, 1e30, shutter_t01, cone_h);
                        // Rec.709 luminance flattens RGB transmittance
                        // to one scalar. For pure shadow (no glass
                        // between hit and sun), R==G==B so the choice
                        // of weight doesn't matter; the weighting only
                        // kicks in with chromatic dielectrics on the
                        // shadow ray.
                        float vis_s = dot(t_vis,
                                          float3(0.2126, 0.7152, 0.0722));
                        frame_shadow_vis += saturate(vis_s);
                    } else {
                        frame_shadow_vis += 1.0;   // sun below horizon
                    }
                } else {
                    frame_shadow_vis += 1.0;       // sky pixel
                }
            }
        }
#if PT_WATER_ENABLED
        // --- Underwater medium tracking (Phase 4, #134) ---------------------
        // Apply Beer's-law attenuation over the segment we just traced. The
        // ray travelled distance `h.t` through whatever medium `in_water`
        // currently identifies (set by ray-gen, flipped only by the
        // MAT_WATER refract path -- MAT_DIELECTRIC is intentionally left
        // out per the Phase 4 simplification, see the in_water docstring
        // at the bounce-loop preamble). This replaces the old per-bounce
        // nominal 1.5m estimate at MAT_WATER ingress with the exact path
        // length, and also tints the camera-underwater primary segment
        // (subsuming the b == 0 special case from b2690ac).
        //
        // Misses (h.hit == false) skip this: in_water + miss is caught
        // by the deep-water tint fallback below.
        // --- #305: cash the water-leaving term now that the column has a
        // LENGTH. The source is set at the interface (see the MAT_WATER
        // refract branch), where the geometry is known but the depth is
        // not; it is spent here, where the trace has just measured how
        // much water the ray actually crossed. A bottom hit truncates the
        // integral at exactly the right place, so shallow water over sand
        // and open ocean are the same expression -- there is no deep-water
        // branch to disagree with a shallow one at the transition.
        //
        // BEFORE the segment Beer's law, deliberately: wl_amp already
        // carries the eye-transmittance exp(-sigma_t t) inside its own
        // depth integral, so the throughput this multiplies is the one at
        // the interface, not the one at the far end of the column.
        if (wl_pending && ptMediumIsWater(medium)) {
            float3 sat = h.hit ? (float3(1.0) - exp(-wl_rate * max(h.t, 0.0)))
                               : float3(1.0);
            float3 wl_c = throughput * wl_amp * sat;
            radiance += wl_primary ? wl_c : clampIndirect(wl_c);
            wl_pending = false;
        }
        if (ptMediumIsWater(medium) && h.hit) {
            float3 absorb = water_params0.xyz;
            throughput *= exp(-absorb * h.t);
        }
        // --- end Underwater medium tracking ---------------------------------
#endif  // PT_WATER_ENABLED
        if (!h.hit) {
#if PT_WATER_ENABLED
            // --- Underwater sky-miss override (Phase 4, #134) ---------------
            // Ray escaped to infinity while inside the water medium. We do
            // NOT call skyColor(ro, rd, seed) here -- a TIR-reflected
            // underwater ray
            // looking back down at empty space should die to deep-water
            // void, not pick up bright sky-blue from the procedural sky
            // model. water_state.yzw is the tunable deep-water tint
            // (r_water_deep_color_r/g/b, defaults near-black). Multiply
            // by throughput so segment absorption + Fresnel attenuations
            // along the way also dim the deep tint -- a TIR ray with a
            // long underwater path before escaping fades to black.
            if (ptMediumIsWater(medium)) {
                if (b == 0u) {
                    radiance += throughput * water_state.yzw;
                } else {
                    radiance += clampIndirect(throughput * water_state.yzw);
                }
                break;
            }
            // --- end Underwater sky-miss override ---------------------------
#endif  // PT_WATER_ENABLED
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
                    // origin (offset by ptRayOrigin from the surface).
                    float pdf_env = pdfEnvMapDir(rd);
                    float w_brdf  = misBalance(prev_lambert_brdf_pdf, pdf_env);
                    float3 trans  = transmittance(ro, rd, 1e30, shutter_t01,
                                                  cone);
                    radiance += clampIndirect(throughput * skyColor(ro, rd, seed) * trans * w_brdf);
                }
            } else if (!skip_sky_on_miss) {
                // b == 0 is the camera ray hitting the sky directly --
                // unbounded so the HDRI sun renders at full intensity.
                // b > 0 is a bounce ray escaping; clamp those to kill
                // fireflies from BSDF-sampled bounces hitting the sun.
                if (b == 0u) {
                    // Celestials composite (issue #46): when the
                    // denoiser is on and the user opted in
                    // (r_star_split 1, default on), peel
                    //   starsOnly  (sub-pixel BSC / procedural stars)
                    //   sunDisc    (0.55 deg disc + halo)
                    //   moonDisc   (0.55 deg disc + textured surface)
                    // out of the primary-miss sky term so denoise_color
                    // reaches SVGF/MetalFX as just the procedural sky
                    // base + NEE indirect contributions. The engine's
                    // StarsComposite kernel (dispatched post-Denoise,
                    // pre-bloom) re-adds the three terms per pixel with
                    // N aperture-sampled rays so:
                    //   - stars stay sub-pixel sharp (no a-trous smudge
                    //     across a 17x17 neighbourhood)
                    //   - sun/moon get a clean bokeh disc in ONE frame
                    //     (instead of accumulating noisy 1-spp aperture
                    //     samples through the denoiser's temporal stage)
                    //
                    // Procedural sky mode only (engine gate); HDRI mode
                    // bakes celestials into env_map (can't separate
                    // without a brightness threshold) and starsOnly /
                    // sunDisc / moonDisc all early-out on the gradient
                    // / hdri modes anyway. Bounce rays (b > 0) keep
                    // their existing path: clampIndirect kills firefly
                    // contributions from sub-pixel stars on secondary
                    // bounces, and the indirect contribution to NEE
                    // light off the moon / sun rests on the
                    // hdri_lights[] / sun_rad NEE paths (not skyColor
                    // sampling). So secondary illumination from stars,
                    // moon and sun is preserved.
                    float3 sky_full = skyColor(ro, rd, seed);
                    if (composite_celestials_active) {
                        // Subtract every celestial term skyColor()
                        // injects in mode==2:
                        //   starsAtPrimaryMiss(rd)        -- BSC / procedural starfield
                        //   sunDisc(rd, sun)              -- 0.55deg sun disc + atmospheric halo
                        //   moonDisc(rd, moon, sun)       -- moon disc + textured surface + halo
                        //   procSkyInlineSunDiscAndHalo() -- procSky's own inline sun
                        //                                    (factored out into a helper so the
                        //                                    composite + this subtraction stay
                        //                                    a single source of truth)
                        // The disc helpers self-gate on sun/moon elevation,
                        // and starsAtPrimaryMiss returns 0 outside
                        // procedural-sky mode, so calling all of them on
                        // the gated branch is safe (no-op when out of view).
                        float3 stars_only = starsAtPrimaryMiss(rd);
                        float3 sun_d      = sunDisc(rd, sun_and_mode.xyz);
                        float3 moon_d     = moonDisc(rd, moon_dir_phase,
                                                     sun_and_mode.xyz);
                        float3 proc_sun   = procSkyInlineSunDiscAndHalo(
                                                rd, sun_and_mode.xyz);
                        radiance += throughput *
                                    (sky_full - stars_only - sun_d - moon_d - proc_sun);
                    } else {
                        radiance += throughput * sky_full;
                    }
                } else {
                    // Planetary P5 (#259): a bounce ray launched by a
                    // planetary-ocean specular hit sees the sky WITHOUT its
                    // sun disc -- that lobe already integrated the sun with
                    // a shadow ray. Every other bounce is unchanged.
                    radiance += clampIndirect(throughput *
                        (ocean_sun_nee_prev ? skyColorNoSunDisc(ro, rd, seed)
                                            : skyColor(ro, rd, seed)));
                }
            }
            break;
        }

        float3 hit_pt = ro + rd * h.t;

        // Ray-cone mip LOD (#26 follow-up): widen the cone by this segment's
        // length, then project onto the surface for a fractional mip level.
        // Must run BEFORE applyPbrTextures so the trilinear fetch is filtered.
        // Grazing/far hits (the receding checkerboard horizon) get a coarse
        // LOD, which is what removes the far-field texture aliasing.
        cone_width += cone_spread * h.t;
        h.tex_lod = pbrConeTexLod(cone_width, rd, h.normal, h.uv_scale);
        // ...and the same footprint is what every secondary ray leaving
        // this hit uses for its epsilons (#256). One object, two consumers:
        // the mip level the texture is read at, and the distance a ray has
        // to clear the surface by. They agree by construction, which is the
        // property an absolute 1e-3 could not have.
        PtRayCone cone_hit = ptConeMake(cone_width, cone_spread);

        // Wave 8 PBR (#26): apply the hit material's texture maps in place
        // BEFORE deriving the front-facing normal (so a normal map feeds
        // nf) and BEFORE the BSDF dispatch (so every material branch sees
        // textured albedo / roughness / metallic). No-op + bit-identical
        // when the material has no texture tiles (the default), because
        // applyPbrTextures gates each sample on tile != kPbrNoTexTile.
        applyPbrTextures(h);

        // Wave 9 advanced materials (#wave9): resolve any per-prim BSDF
        // lobe extension (anisotropic / clearcoat / subsurface) from the
        // push-constant override array by prim id. No-op + bit-identical
        // when the scene has no advanced materials (mat_override_count.x ==
        // 0 short-circuits inside resolveMatExt).
        resolveMatExt(h);

        float  ndi = dot(h.normal, rd);
        float3 nf  = (ndi < 0.0) ? h.normal : -h.normal;

        // Per-prim emission (#181 polish). h.emission is sourced from
        // primitives[idx*7 + 4].xyz in testAnalyticPrim, units W/sr per
        // channel. Adds `throughput * emission` BEFORE the BRDF eval so
        // the body itself glows AND illuminates nearby surfaces via
        // bounce-sampled rays that land on it. Camera-ray (b==0) hits
        // get full unclamped emission; bounce-ray (b>0) hits go through
        // clampIndirect to kill firefly contributions from tiny bright
        // emitters that bounce rays happen to land on after a Lambert
        // surface (the high-variance estimator). Zero emission is the
        // default; non-emissive prims pay just one float3 add per hit.
        if (any(h.emission > float3(0.0))) {
            if (b == 0u) {
                radiance += throughput * h.emission;
            } else {
                radiance += clampIndirect(throughput * h.emission);
            }
        }

        // Editor backend (agent-19): magenta silhouette outline around
        // the currently-selected analytic primitive. Only paints on the
        // primary ray (b == 0) so accumulated bounces don't smear the
        // outline through the scene; only on hits whose user-facing
        // prim_id matches editor_params.x (the host writes the
        // EFFECTIVE id there: 0 when the gate is off OR kind !=
        // AnalyticPrim, so the != 0 check is enough). The silhouette
        // condition `dot(h.normal, -rd) < 0.2` keeps the ring at the
        // grazing-angle rim of the body where the surface bends away
        // from the camera; the body interior renders normally so the
        // user still sees their material. Magenta (1, 0, 1) was chosen
        // for high contrast on both bright and dark scenes.
        //
        // Bit-identical when no selection is set OR r_editor_outline is
        // 0: editor_params.x == 0 short-circuits before the per-pixel
        // silhouette dot product, so non-editor frames pay exactly one
        // u32 compare per primary hit -- no measurable cost.
        if (b == 0u && editor_params.x != 0u && h.prim_id == editor_params.x) {
            float rim = saturate(0.2 - dot(h.normal, -rd)) * 5.0;
            if (rim > 0.0) {
                radiance += throughput * float3(8.0, 0.0, 8.0) * rim;
            }
        }

        // --- Wave 9 clearcoat (car paint) lobe (#wave9) -------------------
        // Stochastic layered model: a hit carrying MAT_EXT_CLEARCOAT first
        // tests the clear lacquer coat. With probability p = weight *
        // Fresnel(coat) the ray reflects off the coat (an independent GGX
        // gloss lobe, IOR 1.5) and this bounce is done; otherwise it
        // transmits through the coat and falls through to the base
        // material dispatch below. Selecting the branch WITH probability
        // p and NOT reweighting either side leaves the physical energy
        // split in expectation: coat = p * G2/G1 = weight*F*G2/G1, base
        // = (1 - weight*F) * base_reflectance. (The previous form divided
        // the coat by p and the base by (1-p), cancelling both losses --
        // total reflectance F + base > 1, so a coated white sphere GAINED
        // energy in a furnace test and brightened every bounce.)
        // Bit-identical when the flag is clear (h.mat_ext defaults 0, so
        // this whole block is skipped).
        if ((h.mat_ext & MAT_EXT_CLEARCOAT) != 0u && h.clearcoat_weight > 0.0) {
            float cos_v = clamp(dot(nf, -rd), 0.0, 1.0);
            float fc = clearcoatFresnel(cos_v);
            float p_coat = clamp(h.clearcoat_weight * fc, 0.0, 0.99);
            if (randf(seed) < p_coat) {
                // Reflect off the coat. cs.weight is the VNDF masking
                // ratio G2/G1; the Fresnel * weight energy factor is
                // carried by the selection probability itself (one-
                // sample estimator), so no 1/p reweight -- that's what
                // keeps the layer energy-conserving.
                CoatSample cs = sampleClearcoat(nf, -rd, h.clearcoat_rough, seed);
                if (!cs.valid) { break; }
                throughput *= cs.weight;
                ro = ptRayOrigin(hit_pt, nf, cone_hit, cs.dir);
                rd = cs.dir;
                skip_sky_on_miss = false;
                prev_was_lambert_nee = false;
                // Russian roulette (mirror the per-material tail below so a
                // continue here doesn't skip path termination).
                if (b >= 2u) {
                    float pr = max(throughput.r, max(throughput.g, throughput.b));
                    pr = clamp(pr, 0.05, 1.0);
                    if (randf(seed) > pr) break;
                    throughput /= pr;
                }
                continue;
            }
            // Coat transmission: NO reweight. The (1 - p_coat) selection
            // probability IS the coat's physical transmission loss in
            // expectation -- dividing it back out (the old form) handed
            // the base its full unattenuated energy on top of the coat's
            // Fresnel reflection.
        }
        // --- end Wave 9 clearcoat lobe ------------------------------------

        // --- Wave 9 subsurface scattering (#wave9) ------------------------
        // A hit carrying MAT_EXT_SUBSURFACE runs a bounded random walk
        // through the medium INSTEAD of the base BSDF dispatch: the ray
        // transmits into the surface, scatters off an HG phase function with
        // per-channel single-scatter albedo until it either exits the body
        // (resume the bounce loop in air) or is absorbed deep inside
        // (terminate). This gives the soft, depth-tinted translucency of
        // skin / wax / marble. Bit-identical when the flag is clear (the
        // block is skipped and the base material shades as before).
        if ((h.mat_ext & MAT_EXT_SUBSURFACE) != 0u) {
            SssWalk sw = subsurfaceRandomWalk(hit_pt, rd, h.normal,
                                              h.subsurface_radius,
                                              h.subsurface_color,
                                              shutter_t01,
                                              ptConeMake(cone_width, cone_spread),
                                              seed);
            throughput *= sw.tint;
            if (!sw.escaped) { break; }   // absorbed deep in the medium
            ro = sw.ro;
            rd = sw.rd;
            skip_sky_on_miss = false;
            prev_was_lambert_nee = false;
            // Russian roulette (mirror the per-material tail so a continue
            // here doesn't skip path termination).
            if (b >= 2u) {
                float pr = max(throughput.r, max(throughput.g, throughput.b));
                pr = clamp(pr, 0.05, 1.0);
                if (randf(seed) > pr) break;
                throughput /= pr;
            }
            continue;
        }
        // --- end Wave 9 subsurface scattering -----------------------------

        // ReSTIR: every primary-hit pixel must (re)write its reservoir slot
        // each frame. PathTrace only builds a reservoir inside the
        // MAT_LAMBERT branch below, so a non-Lambert primary hit (metal /
        // dielectric / water) would leave this pixel's slot holding a prior
        // frame's reservoir from the ping-pong buffer -- which RestirFinal
        // then shades as phantom diffuse light on the chrome/glass surface.
        // Stamp an empty reservoir here first; the MAT_LAMBERT branch
        // overwrites it with the real one when it builds it.
        if ((restir_enabled != 0u) && (b == 0u) && h.mat != MAT_LAMBERT) {
            uint pixel_idx = tid.y * dim.x + tid.x;
            storeReservoir(reservoir_curr_buf, pixel_idx, initReservoir());
        }

        if (h.mat == MAT_LAMBERT) {
            // --- Analytic light primitives NEE (#73 / ReSTIR #78) -------
            // Direct-lighting from the engine's analytic light list:
            // point / spot / sphere area / quad area. Two code paths
            // share this branch:
            //   r_restir = 0 : legacy single-pick uniform NEE (one
            //       random light, one shadow ray, contribution added
            //       here). Variance bounded for ~dozen lights;
            //       beyond that the user should turn ReSTIR on.
            //   r_restir = 1 (Phase A): WRS candidate generation at
            //       the PRIMARY hit (b == 0) only. Builds a per-pixel
            //       reservoir of K candidates, writes it to
            //       reservoir_curr_buf, and DOES NOT add a
            //       contribution here -- the RestirFinal kernel
            //       additively blends the shadow-tested resampled
            //       sample into denoise_color after the temporal +
            //       spatial reuse passes run. Indirect bounces
            //       (b > 0) keep the legacy single-pick NEE because
            //       Phase A is direct-illumination only (#78 Phase B
            //       extends to GI bounce reservoirs).
            bool restir_active = (restir_enabled != 0u) && (b == 0u);
            if (light_count > 0u && !restir_active) {
                LightSample ls = sampleAnalyticLight(hit_pt, nf, seed);
                float n_dot_l = dot(nf, ls.dir);
                if (ls.pdf > 0.0 && n_dot_l > 0.0) {
                    // Shadow ray: surface to sampled point on the
                    // light. Stop t_max just before the light so the
                    // light's own surface doesn't self-block.
                    // Stop t_max one near-cut short of the light, for the
                    // same reason the origin is offset by one (#256): the
                    // light's own surface must not self-block, and "its own
                    // surface" is a scale-relative distance, not a mm.
                    float3 sro = ptRayOrigin(hit_pt, nf, cone_hit, ls.dir);
                    float t_max = max(ls.dist - ptRayTMin(sro, cone_hit), 0.0);
                    float3 trans = transmittance(sro,
                                                 ls.dir, t_max, shutter_t01,
                                                 cone_hit, medium);
                    if (any(trans > float3(0.0))) {
                        float3 brdf = h.albedo / 3.14159265358979;
                        float3 c = throughput * ls.Le * trans * brdf *
                                   n_dot_l / ls.pdf;
                        // Direct light at the primary hit is a LOW-
                        // variance estimate -- clamping it dims bright
                        // close lights ~2.5x and visibly flattens the
                        // inverse-square falloff (the procedural sun
                        // NEE is deliberately unclamped for the same
                        // reason). The clamp earns its keep only on
                        // the high-variance indirect bounces.
                        radiance += (b == 0u) ? c : clampIndirect(c);
                    }
                }
            }
            if (light_count > 0u && restir_active) {
                // --- ReSTIR DI Phase A: WRS candidate generation ----
                // Build a per-pixel reservoir over K candidates. Each
                // candidate is one uniform-random pick from the
                // light list + per-type area sample; the WRS picks
                // ONE survivor weighted by its target pdf (un-shadowed
                // Lambert contribution). No shadow rays here -- the
                // RestirFinal kernel does the SINGLE survivor shadow
                // ray after temporal + spatial reuse. This is the
                // whole point of ReSTIR: replace K shadow rays with
                // one.
                Reservoir res = initReservoir();
                uint K = min(restir_k_candidates, kRestirMaxK);
                if (K < 1u) K = 1u;
                for (uint k = 0u; k < K; ++k) {
                    LightSample ls = sampleAnalyticLight(hit_pt, nf, seed);
                    float n_dot_l = dot(nf, ls.dir);
                    if (ls.pdf <= 0.0 || n_dot_l <= 0.0) {
                        // Empty candidate -- skip the streamingUpdate's
                        // M increment so back-facing samples don't
                        // depress W for legitimate ones. We do
                        // increment M though, because Bitterli's
                        // bias correction needs the total candidate
                        // count, not just non-zero-weight ones.
                        res.M += 1u;
                        continue;
                    }
                    // Target pdf (un-shadowed Lambert estimate). Note
                    // that sampleAnalyticLight pre-bakes 1/dist^2 +
                    // cos_light + spot falloff into ls.Le for the
                    // POINT / SPOT / SPHERE / QUAD types, so the
                    // target pdf reduces to the receiver-side n_dot_l
                    // times a luminance reduction on ls.Le. brdf
                    // (albedo/PI) is *constant* across all candidates
                    // for a Lambert surface, so we omit it from the
                    // target pdf — it cancels in W during
                    // finalizeReservoir.
                    float3 contrib = ls.Le * n_dot_l;
                    float p_hat = max(dot(contrib,
                                          float3(0.2126, 0.7152, 0.0722)),
                                      0.0);
                    float u = randf(seed);
                    if (streamingUpdate(res, p_hat, ls.pdf, u)) {
                        // Cache the surviving sample's geometry for
                        // the spatial / temporal merge passes.
                        res.light_pos = hit_pt + ls.dir * ls.dist;
                        // ls.dir points from hit toward light; the
                        // light surface normal at the chosen sample
                        // point is what receivers' merge passes need
                        // to evaluate cos_light. For point/spot delta
                        // lights we store the inverse light direction
                        // (so dot(light_n, -wi) = 1 trivially); for
                        // SPHERE / QUAD the merge passes consult this
                        // when re-evaluating the survivor's pdf at a
                        // different receiver. We approximate by
                        // storing -ls.dir for non-area, which is
                        // exact for point lights and close enough
                        // for spot lights at MVP scope.
                        res.light_n   = -ls.dir;
                        res.Le        = ls.Le;
                        res.pdf_solid = ls.pdf;
                        res.dist      = ls.dist;
                        // light_idx: the entry the SURVIVOR was
                        // actually sampled from, threaded out of
                        // sampleAnalyticLight. The previous form drew
                        // a fresh independent uniform index here --
                        // wrong with probability (N-1)/N -- so the
                        // temporal/spatial merge passes recomputed
                        // p_hat (and light type / intensity / spot
                        // params) against an unrelated light and the
                        // reuse chain resampled garbage.
                        res.light_idx = ls.light_idx;
                    }
                }
                finalizeReservoir(res);
                uint pixel_idx = tid.y * dim.x + tid.x;
                storeReservoir(reservoir_curr_buf, pixel_idx, res);
                // NO contribution added to radiance here — RestirFinal
                // adds it after the reuse chain has finished resampling.
            } else if (restir_active) {
                // r_restir on, but light_count == 0 — write an empty
                // reservoir so the chain's reads don't see stale
                // data from a previous frame's light set. Cheap (4
                // float4 writes); avoids an additional cvar-tied
                // dispatch toggle on the reservoir buffer clear.
                Reservoir res = initReservoir();
                uint pixel_idx = tid.y * dim.x + tid.x;
                storeReservoir(reservoir_curr_buf, pixel_idx, res);
            }
            // --- end Analytic light NEE / ReSTIR -----------------------
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
                    float3 trans = transmittance(ptRayOrigin(hit_pt, nf, cone_hit, env.dir),
                                                 env.dir, 1e30, shutter_t01,
                                                 cone_hit, medium);
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
                        float3 t_s = transmittance(ptRayOrigin(hit_pt, nf, cone_hit, hl.dir),
                                                   hl.dir, 1e30, shutter_t01,
                                                   cone_hit, medium);
                        if (any(t_s > float3(0.0))) {
                            float3 brdf = h.albedo / 3.14159265358979;
                            float3 c = throughput * hl.color * t_s *
                                       brdf * n_dot_l / hl.pdf;
                            // Same convention as the analytic-light NEE:
                            // direct light at the primary hit stays
                            // unclamped (low variance; clamping dims
                            // bright sun clusters), indirect bounces
                            // keep the firefly clamp.
                            radiance += (b == 0u) ? c : clampIndirect(c);
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
                // NO separate diffuse "ambient" tap here. Env-NEE above
                // (importance-sampled from the HDRI CDF) plus the MIS-
                // weighted bounce-ray sky-miss complement (or, with MIS
                // off, env-NEE alone with the miss suppressed) already
                // form a COMPLETE estimator of the diffuse environment
                // integral. A previous cosine-hemisphere "ambient" tap
                // here re-integrated the same integral a second time --
                // a clean 2x over-count of diffuse env light (white-
                // furnace: albedo-1 Lambert under a uniform L=1 map
                // returned exitant 2.0). The two RNG draws it consumed
                // are gone with it; HDRI goldens regenerate darker by
                // the duplicated diffuse term.
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
                // Planetary P0 (#254): sin(solar elevation) AT THE HIT
                // POINT. This is the site that makes a terminator
                // possible -- once the frame is spherical, two surfaces a
                // quarter-planet apart get different sun elevations and
                // one of them is in night. Planar frame -> sun_dir.y
                // bit-for-bit.
                float  sun_elev = elevationAt(hit_pt, sun_dir);

                if (sun_elev > 0.0) {
                    float n_dot_l = max(0.0, dot(nf, sun_dir));
                    float3 sun_jit = sampleSunDisc(sun_dir, seed);
                    float3 trans  = (n_dot_l > 0.0)
                        ? transmittance(ptRayOrigin(hit_pt, nf, cone_hit, sun_jit),
                                        sun_jit, 1e30,
                                        shutter_t01, cone_hit, medium)
                        : float3(0.0);
                    // Atmospheric attenuation: optical depth times
                    // air mass. Air mass = 1/sin(elev) (thin-slab
                    // approximation, accurate down to ~5 deg). The
                    // exp(-tau*am) curve matches real sunset
                    // behaviour: brightness drops off a cliff once
                    // elev < 5 deg, but stays bright through midday.
                    //
                    // HOISTED out of the any(trans>0) visibility gate for
                    // the B10 SIGMA-demod capture below: sun_rad depends
                    // only on sun elevation (not on the shadow ray), so
                    // computing it for a fully-shadowed pixel is a few
                    // wasted ALU ops, never a different radiance value.
                    // Every expression here is byte-identical to the
                    // pre-fix code; only the scope changed, so the
                    // `radiance +=` below is bit-for-bit unchanged.
                    // Sunlight arriving at the hit point. With
                    // r_sun_physical_transmittance off (the default)
                    // this is byte-identical to the pre-#257 tuned
                    // curve; with it on, the reddening comes from the
                    // real slant-path optical depth through Rayleigh,
                    // aerosol and ozone instead of from four
                    // hand-authored constants. See sunIrradianceAt.
                    float3 sun_rad = sunIrradianceAt(hit_pt, sun_dir, sun_elev);
                    float3 brdf    = h.albedo / 3.14159265358979;
                    if (n_dot_l > 0.0 && any(trans > float3(0.0))) {
                        // Multiply by per-channel transmittance through
                        // any dielectrics on the shadow ray, so glass
                        // tint / diamond focal lensing modulate the
                        // direct light correctly.
                        float3 sun_contrib =
                            throughput * sun_rad * trans * brdf * n_dot_l;
                        radiance += sun_contrib;
                        // B10 SIGMA demod: record the SHADOWED direct-sun
                        // value actually folded into radiance at the
                        // primary hit -- subtracted from the denoiser
                        // input at frame end so SVGF never blurs the hard
                        // sun-shadow edge. Same value bit-for-bit as the
                        // `radiance +=` above.
                        if (b == 0u && write_shadow_vis != 0u) {
                            sample_sun_shadowed = sun_contrib;
                        }
                    }
                    // B10 SIGMA demod: record the UNSHADOWED direct-sun
                    // radiance S (independent of the shadow ray) so the
                    // post-denoise composite can re-modulate it by the
                    // separately-denoised visibility. Captured whenever the
                    // sun is up and front-facing, even when fully shadowed
                    // this sample (so a penumbra pixel still has a valid S
                    // for the bilateral re-modulation).
                    if (b == 0u && write_shadow_vis != 0u && n_dot_l > 0.0) {
                        sample_sun_direct = throughput * sun_rad * brdf * n_dot_l;
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
                            float3 t_m = transmittance(ptRayOrigin(hit_pt, nf, cone_hit, moon_jit),
                                                       moon_jit, 1e30, shutter_t01,
                                                       cone_hit, medium);
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
                // Calculated diffuse skylight ambient. Sample the sky
                // BASE dome at the surface normal -- single-direction
                // Lambert ambient approximation, real and derived from
                // the sky model rather than hardcoded RGB. Deliberately
                // skyColorBase, not skyColor: the full sky includes the
                // sun disc + up-to-8-degree horizon halo and the moon
                // disc, and a normal pointing near a celestial body
                // would receive its direct radiance as unshadowed
                // "ambient" ON TOP of the explicit sun/moon NEE above.
                float3 ambient = skyColorBase(hit_pt, nf, seed);
#if PT_WATER_ENABLED
                // --- Planetary P7 (#261): the glowing sea floor ------------
                //
                // THE BUG. This term was never gated on the medium, so a
                // submerged Lambert surface received the FULL above-water
                // sky dome -- no Beer-Lambert for the column above it, no
                // Fresnel loss at the interface, no shadow ray at all. A
                // seabed at 100 m was lit exactly as brightly as a beach.
                // It is the first thing anyone notices underwater and it
                // was named as such in #259 and #261 both.
                //
                // THE FIX IS NOT A FUDGE FACTOR. transmittance() already
                // computes precisely the quantity that is missing: it
                // marches the medium stack, applies exp(-a * segment) over
                // every water leg, folds (1 - F) in at the interface, and
                // returns zero on total internal reflection. So the
                // ambient becomes the skylight that ACTUALLY REACHED this
                // point, by the same estimator every other NEE site here
                // uses, rather than the skylight that would have reached it
                // in air.
                //
                // WHAT IS STILL AN APPROXIMATION, stated rather than
                // hidden: the DIRECTION stays nf. This is a
                // single-direction hemispherical-ambient approximation --
                // it was one before this change too -- so the sky is
                // sampled along the unrefracted normal while the
                // MAGNITUDE now follows the refracted path. Making the
                // direction right as well needs transmittance() to hand
                // back its exit direction, which changes a signature six
                // call sites share. The magnitude is what "no glowing sea
                // floor" is about.
                //
                // Costs nothing above water: the branch is one predicate on
                // a register the loop already carries.
                if (ptMediumIsWater(medium)) {
                    ambient *= transmittance(
                        ptRayOrigin(hit_pt, nf, cone_hit, nf), nf, 1e30,
                        shutter_t01, cone_hit, medium);
                }
                // --- end Planetary P7 ---------------------------------------
#endif
                radiance += throughput * h.albedo * ambient;
                skip_sky_on_miss = true;
            }
            throughput *= h.albedo;
            rd = sampleCosineHemisphere(nf, seed);
            ro = ptRayOrigin(hit_pt, nf, cone_hit, rd);
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
            // Wave 9 (#wave9): anisotropic GGX (brushed metal) when the
            // prim carries the MAT_EXT_ANISOTROPIC flag. The base metal F0
            // is the albedo (a metal's specular reflectance IS its colour),
            // so a VNDF-sampled anisotropic lobe with Fresnel folded in
            // gives a physically-grounded stretched highlight. The default
            // (flag clear) keeps the cheap isotropic roughness-perturbed
            // mirror bit-for-bit, so existing metal goldens are unaffected.
            if ((h.mat_ext & MAT_EXT_ANISOTROPIC) != 0u) {
                AnisoSample asmp = sampleAnisoGgx(nf, -rd, h.albedo,
                                                  h.roughness, h.aniso_amount,
                                                  h.aniso_rotation, seed);
                if (!asmp.valid) { break; }   // sampled below the horizon
                throughput *= asmp.weight;
                ro = ptRayOrigin(hit_pt, nf, cone_hit, asmp.dir);
                rd = asmp.dir;
            } else {
                float3 r = reflect(rd, nf);
                r = normalize(r + h.roughness * (sampleCosineHemisphere(nf, seed) - nf));
                throughput *= h.albedo;
                ro = ptRayOrigin(hit_pt, nf, cone_hit, r);
                rd = r;
            }
            skip_sky_on_miss = false;   // metal didn't NEE; sky on miss is real
            prev_was_lambert_nee = false;   // specular bounce: MIS doesn't apply here
        }
        else if (h.mat == MAT_DIELECTRIC) {
            // --- Wave 8 spectral (#27) ---
            // Wavelength-dependent index of refraction. When spectral mode
            // is off (hero_lambda < 0), mat_ior == h.ior exactly so this
            // branch is bit-identical to the pre-#27 achromatic path. When
            // on, the material's nominal IOR (h.ior, taken as the index at
            // the d-line) anchors a real Cauchy curve and the hero
            // wavelength reads its own index off it -- shorter wavelengths
            // get a higher index and bend more, which is dispersion. The
            // default Cauchy C (spectral_params.y, BK7 crown glass) governs
            // the spread; the host could later source a per-material Abbe
            // number via abbeToCauchyC().
            float mat_ior = h.ior;
            if (hero_lambda >= 0.0) {
                mat_ior = cauchyIOR(h.ior, spectral_params.y, hero_lambda);
            }
            // --- end ---
            // Planetary P7 (#261) leaves this at 1.0 -- see the MAT_WATER
            // branch below for the measurement that decided it.
            float ior_a = (ndi < 0.0) ? 1.0     : mat_ior;
            float ior_b = (ndi < 0.0) ? mat_ior : 1.0;
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
                ro = ptRayOrigin(hit_pt, nf, cone_hit, rd);
                // Fresnel REFLECTION (incl. TIR) off the glass. A Lambert
                // env-NEE's shadow ray follows ONLY the transmitted lobe
                // (transmittance() keeps refraction, never the reflected
                // lobe), so this reflected path is a higher-order term no
                // prior NEE sampled -- its eventual sky-on-miss is genuinely
                // new light. Clear the specular-suffix suppression so the
                // miss is added at full weight.
                skip_sky_on_miss = false;
                prev_was_lambert_nee = false;
            } else {
                rd = normalize(refracted);
                ro = ptRayOrigin(hit_pt, -nf, cone_hit, rd);
#if PT_WATER_ENABLED
                // Planetary P7 (#261): glass joins the medium stack. It was
                // deliberately excluded from the old one-bit flip, which is
                // why a glass body submerged in water kept accumulating
                // WATER absorption over the segment inside the glass. Above
                // water this only moves the stack between "empty" and "one
                // dielectric" and nothing reads it in that state.
                medium = (ndi < 0.0)
                             ? ptMediumEnter(medium, MEDIUM_DIELECTRIC)
                             : ptMediumPop(medium);
#endif
                // REFRACTION (transmission). If a preceding Lambert env-NEE
                // already accounted for the sky arriving through this same
                // dielectric (its shadow ray refracts through it via
                // transmittance(), returning a non-zero tint), then adding
                // the sky-on-miss down this transmitted lobe would DOUBLE-
                // COUNT that light -- the very bug this branch used to have,
                // because it reset the flags unconditionally and the single-
                // segment MIS/skip bookkeeping was lost after one bounce.
                // PERSIST skip_sky_on_miss / prev_was_lambert_nee unchanged
                // so the eventual miss is suppressed (MIS off) or balance-
                // weighted (MIS on) exactly as the single-segment case does,
                // now carried across the whole refraction chain. When no
                // Lambert NEE preceded (e.g. a camera ray hitting a bare
                // glass body), both flags are already false, so refracting
                // the sky still shows it -- this branch is a no-op there.
            }
        }
#if PT_WATER_ENABLED
        else if (h.mat == MAT_WATER) {  // (#134)
            // Water material: shading-normal-perturbed analytic plane with
            // Schlick Fresnel (F0=0.02), Snell refraction via the refract()
            // builtin, and Beer's-law absorption applied per-segment over
            // the bounce loop (see the segment-absorption block at the
            // top of the loop body and water_state at the bounce-loop
            // preamble). No NEE on the water surface itself -- the
            // reflected / refracted ray inherits the current throughput
            // and the next bounce handles direct lighting through
            // whichever medium the ray entered.
            //
            // Perturbed shading normal.  Geometric ndi-driven sign was
            // computed off h.normal (geometric); use it to orient the
            // shading normal toward the incoming ray, so under-water hits
            // (sphere submerged) and above-water hits share one branch.
            float water_ior = clamp(h.ior > 0.0 ? h.ior : water_params0.w,
                                    1.0, 2.4);
            // FFT ocean (#25): when on, h.normal already carries the
            // displaced FFT surface normal from the heightfield march --
            // use it directly instead of re-perturbing with the Phase 1
            // value-noise wave normal (which would double-bump the
            // surface). Off -> the Phase 1 analytic wave normal as before.
            // Planetary P5 (#259) adds the second disjunct: the Phase 1
            // wave normal is a value-noise field in world XZ, which on a
            // sphere is a horizontal-plane pattern wrapped round a globe.
            // A planetary shell takes its macro normal even when the FFT is
            // off -- that is the smooth Cox-Munk sphere the handover is
            // measured against, and it must not have ripples painted on it.
            float3 shading_n = (ocean_params0.x != 0.0 || planet_ocean.x > 0.0)
                                   ? h.normal
                                   : waterShadingNormal(hit_pt, h.normal);
            // --- Wave 9 ocean-foam ------------------------------------------
            // Foam: where the choppy displacement folds (crest breaking) OR a
            // wind-driven whitecap rides a crest, blend the transparent-water
            // BRDF toward a bright, rough Lambertian froth. The CPU solver
            // bakes crest-fold + wind-coverage + persistence into
            // ocean_displacement.w (-> ocean_foam_here); here we map that raw
            // coverage through a threshold+smoothstep into a perceptual foam
            // mask, then stochastically pick foam-vs-water weighted by the
            // mask. The smoothstep keeps foam off the clean flanks (low
            // coverage reads as 0) while letting it bloom on crests and in
            // choppy troughs; foam_amount (ocean_foam_params.x) widens the
            // lower threshold so cranking intensity spreads broader froth, and
            // 0 skips foam entirely (clean specular sea, A/B-stable).
            float foam_amt = ocean_foam_params.x;
            // Threshold ramp: foam starts at `lo` and saturates at `hi`.
            // Higher foam_amt drops `lo` toward 0 (foam appears earlier ->
            // broader coverage) without ever exceeding `hi`.
            float foam_lo = clamp(0.35 - 0.25 * saturate(foam_amt - 1.0) - 0.30 * min(foam_amt, 1.0), 0.02, 0.35);
            float foam_mask = smoothstep(foam_lo, 0.85, ocean_foam_here);
            foam_mask = saturate(foam_mask * max(foam_amt, 0.0));
            if (ocean_params0.x != 0.0 && foam_amt > 0.0 && foam_mask > 0.0 &&
                randf(seed) < foam_mask) {
                // Bright near-white sea-foam albedo. Real whitecap froth is a
                // dense air-water microbubble scatterer with ~0.9 diffuse
                // reflectance and a faint cool tint (Rayleigh-biased skylight
                // in the bubble interstices). Brighter + whiter than the
                // wave-8 (0.8,0.82,0.85) so breaking crests read as crisp
                // whitecaps against the dark specular sea. The mask already
                // gated coverage, so shade full-strength foam here.
                float3 nf_foam = (dot(h.normal, rd) < 0.0) ? h.normal : -h.normal;
                throughput *= float3(0.92, 0.94, 0.96);   // sea-foam albedo
                // Diffuse/rough froth: cosine-weighted bounce off the FFT
                // normal so the next bounce gathers sky+sun and foam reads as
                // a self-shadowing matte-white surface (no specular highlight,
                // unlike the clean water it overrides).
                rd = sampleCosineHemisphere(nf_foam, seed);
                ro = ptRayOrigin(hit_pt, nf_foam, cone_hit, rd);
                skip_sky_on_miss = false;
                prev_was_lambert_nee = false;
                if (b >= 2u) {
                    float pf = max(throughput.r, max(throughput.g, throughput.b));
                    pf = clamp(pf, 0.05, 1.0);
                    if (randf(seed) > pf) break;
                    throughput /= pf;
                }
                continue;
            }
            // --- end Wave 9 ocean-foam --------------------------------------
            float ndi_s = dot(shading_n, rd);
            float3 nf_s = (ndi_s < 0.0) ? shading_n : -shading_n;

            // --- Planetary P5 (#259): sun glitter, by NEE not by luck -----
            //
            // "Sun glitter off a wave field is the single most recognisable
            // feature of an ocean in a satellite photo" is the reason this
            // phase exists, and it is exactly the thing a purely stochastic
            // specular estimator cannot deliver. The sun disc subtends
            // 0.0093 rad; at the far-field alpha of 0.254 the chance that a
            // VNDF sample lands on it is (0.00465 / 0.254)^2 = 3.4e-4, so at
            // 192 samples per pixel a from-orbit frame gets 0.065 hits --
            // and r_firefly_clamp then bites the few that land. MEASURED:
            // the ocean from 400 km renders with no glint at all.
            //
            // So integrate the sun directly: one shadow ray, the GGX BRDF
            // evaluated toward a jittered point on the disc, and the disc
            // taken out of the bounce ray's sky (see skyColorNoSunDisc,
            // where the two estimators are shown to partition the same
            // light exactly in physical-sky mode).
            //
            // Planetary water only. The flat MAT_WATER planes keep the Wave
            // 8 estimator untouched, and MAT_METAL keeps its own -- widening
            // this to every specular surface is a real improvement and a
            // different change.
            bool did_ocean_sun_nee = false;
            // #305: the sun's visibility from this very point, cached. The
            // water-leaving term below needs exactly the occlusion this
            // block is about to measure, and a second shadow ray toward a
            // direction 0.53 deg away would buy nothing at the price of
            // doubling the ray count on a frame that is mostly ocean.
            // Negative = not measured (the NEE block did not run), which
            // is the signal to fire one.
            float3 ocean_sun_trans = float3(-1.0);
            if (planet_ocean.x > 0.0 && !ptMediumIsWater(medium) &&
                uint(sun_and_mode.w) >= 2u) {
                float3 sun_dir  = sun_and_mode.xyz;
                float  sun_elev = elevationAt(hit_pt, sun_dir);
                float  ndv      = dot(nf_s, -rd);
                if (sun_elev > 0.0 && ndv > 0.0) {
                    // A perfectly smooth interface has a delta lobe and no
                    // NEE to do; the floor keeps D finite there and is the
                    // same 0.02 roughness floor anisoAlpha already clamps to
                    // (0.02^2 = 4e-4).
                    float a  = max(sqrt(ocean_alpha2_here), 4.0e-4);
                    float3 l = sampleSunDisc(sun_dir, seed);
                    float ndl = dot(nf_s, l);
                    float3 hv = normalize(l - rd);
                    float ndh = dot(nf_s, hv);
                    if (ndl > 0.0 && ndh > 0.0) {
                        // Trowbridge-Reitz D, and the same height-correlated
                        // Smith G2 the sampled lobe uses, so the NEE and the
                        // BSDF sample are the same BRDF.
                        float a2 = a * a;
                        float dd = ndh * ndh * (a2 - 1.0) + 1.0;
                        float D  = a2 / (3.14159265358979 * dd * dd);
                        float3 t1, t2;
                        anisoTangentFrame(nf_s, 0.0, t1, t2);
                        float3 vl = float3(dot(-rd, t1), dot(-rd, t2), ndv);
                        float3 ll = float3(dot(l, t1),  dot(l, t2),  ndl);
                        float G2 = ggxG2Aniso(ll, vl, a, a);
                        float F  = fresnelSchlick(max(dot(hv, l), 0.0), 0.02);
                        float fr_spec = D * G2 * F / (4.0 * ndv * ndl);
                        float3 trans = transmittance(
                            ptRayOrigin(hit_pt, nf_s, cone_hit, l), l, 1e30,
                            shutter_t01, cone_hit, medium);
                        ocean_sun_trans = trans;   // #305 reuses this
                        if (any(trans > float3(0.0))) {
                            float3 sun_rad =
                                sunIrradianceAt(hit_pt, sun_dir, sun_elev);
                            float3 contrib =
                                throughput * sun_rad * trans * fr_spec * ndl;
                            radiance += (b == 0u) ? contrib
                                                  : clampIndirect(contrib);
                        }
                        did_ocean_sun_nee = true;
                    }
                }
            }
            // --- Planetary P5 (#259): the other half of the handover ------
            //
            // Everything above this line is geometry. This is where the
            // slope variance the geometry no longer carries comes back as a
            // BRDF: a GGX visible-normal sample about the macro normal with
            // alpha^2 = sigma^2_CoxMunk - sigma^2_resolved(cone). At the
            // boat that alpha is the sub-grid capillary band and the lobe is
            // a near-mirror; from orbit it is the whole Cox-Munk variance
            // and the planet has the broad sun glitter every satellite photo
            // of an ocean shows. No fade, no altitude, no cvar -- only the
            // ray cone, and a conservation law that makes the two halves sum
            // to the same measured constant at every point of the climb.
            //
            // The MEDIUM decision stays on the macro normal. A microfacet
            // normal can point to the other side of the interface, and
            // letting it decide "am I entering or leaving the water" would
            // corrupt in_water for the rest of the path -- the medium is a
            // property of the surface, not of the microfacet.
            //
            // nf_m == nf_s exactly when alpha^2 is 0, which is every
            // non-planetary water surface, so the legacy path is untouched.
            float3 nf_m = nf_s;
            if (ocean_alpha2_here > 0.0) {
                float a_ggx = sqrt(ocean_alpha2_here);
                float3 tt, tb;
                anisoTangentFrame(nf_s, 0.0, tt, tb);
                float3 vl = float3(dot(-rd, tt), dot(-rd, tb), dot(-rd, nf_s));
                if (vl.z > 0.0) {
                    float3 hl = sampleGgxVndfLocal(vl, a_ggx, a_ggx,
                                                   randf(seed), randf(seed));
                    float3 ll = reflect(-vl, hl);
                    // Reject a microfacet whose reflected direction lands
                    // below the MACRO horizon. Such a sample is shadowed by
                    // the surface it came from, and taking it would launch
                    // a ray into the water while the medium bookkeeping
                    // still says air. Falling back to the macro normal is
                    // the conservative answer and it costs nothing at the
                    // small alphas where it almost never fires.
                    if (ll.z > 0.0 && hl.z > 0.0) {
                        nf_m = normalize(hl.x * tt + hl.y * tb + hl.z * nf_s);
                        // VNDF importance sampling leaves the masking ratio
                        // G2/G1 as the surviving weight (Heitz 2018) --
                        // exactly as sampleClearcoat and the anisotropic
                        // metal lobe already do it here.
                        throughput *= ggxG2Aniso(ll, vl, a_ggx, a_ggx) *
                                      (1.0 + ggxLambdaAniso(vl, a_ggx, a_ggx));
                    }
                }
            }
            // --- end Planetary P5 -----------------------------------------

            // --- Planetary P7 (#261): why the outside index is still 1.0 ---
            //
            // The medium stack makes it POSSIBLE to read the far side of
            // this interface off the stack instead of assuming air, and
            // that is the obviously-correct-looking next step. It was
            // implemented, MEASURED, and reverted, so the reasoning is here
            // rather than lost.
            //
            // The measurement first: on the golden matrix it changed
            // NOTHING. golden_water_pool__metal__off -- the one fixture
            // with a submerged body under a wavy surface -- rendered
            // mean-delta 14.313049 against the committed golden both with
            // the stack-derived index and with this literal 1.0, byte for
            // byte the same number. Every other cell was untouched by
            // either. So the change buys no accuracy anything in this repo
            // can see, and it is not free: it makes the refraction depend
            // on the stack, which is topological, where the SIGN of the
            // interface is geometric. Those two disagree on a displaced
            // wave field -- see ptMediumEnter above for the case and for
            // what it cost when the stack update had the same flaw.
            //
            // Doing it properly needs the two notions reconciled: an
            // interface has to know which medium is on each of its sides
            // independently of which way a particular ray met it. That is a
            // change to how the ocean surface reports hits, not a change to
            // this expression, and it is the same problem #133 phase 3's
            // coastal-ribbon water has to solve.
            //
            // So: air outside, exactly as before, and the stack keeps to
            // what it was built for -- what the medium ABSORBS as, and
            // which shadow rays are attenuated -- because that reads the
            // stack alone and never has to agree with a facet normal.
            float ior_a = (ndi_s < 0.0) ? 1.0       : water_ior;
            float ior_b = (ndi_s < 0.0) ? water_ior : 1.0;
            float eta = ior_a / ior_b;
            float cos_i = abs(dot(nf_m, -rd));
            // Schlick F0 for water (1.33 vs air = 0.02041).  Hard-coded
            // here so the cvar can dial ior without the Fresnel split
            // ratio drifting -- F0=0.02 is the perceptually-correct value
            // physically-based renderers use even when ior is mildly
            // perturbed.  fresnelSchlick gives the reflected fraction.
            const float F0_water = 0.02;
            float fr = fresnelSchlick(cos_i, F0_water);
            float3 reflected = reflect(rd, nf_m);
            float3 refracted = refract(rd, nf_m, eta);
            bool tir = (length(refracted) < 1e-3);
            if (tir || randf(seed) < fr) {
                // Reflection (incl. TIR): surface acts like a coated
                // mirror. Ray stays in the SAME medium it was already
                // in, so in_water is unchanged. The next bounce loop
                // iteration will apply per-segment Beer's law if the
                // ray is still underwater.
                rd = reflected;
                ro = ptRayOrigin(hit_pt, nf_s, cone_hit, rd);
                // Reflected lobe: transmittance()-based NEE only follows
                // the transmitted lobe through water, so a reflected
                // sky-on-miss is genuinely new light (mirrors the
                // MAT_DIELECTRIC reflect branch). Clear the suppression.
                skip_sky_on_miss = false;
                prev_was_lambert_nee = false;
                // Planetary P5 (#259): ...except for the sun DISC, which
                // this lobe just integrated by NEE. Set on the REFLECTED
                // lobe only: a transmitted ray leaves through the interface
                // and whatever it eventually sees is a different integral.
                ocean_sun_nee = did_ocean_sun_nee;
            } else {
                // Refraction: the ray crosses the water interface.
                // ndi_s < 0 means the shading normal pointed away from
                // the incoming ray -- we were OUTSIDE the water body
                // and are entering. ndi_s > 0 means we were INSIDE and
                // are exiting back into whatever we were in before.
                //
                // Planetary P7 (#261): push / pop instead of set / clear.
                // "Exiting" used to mean "the medium is now air", which is
                // only true when the water body is not itself inside
                // something. Popping returns to whatever the ray was in,
                // and for the camera-in-air case that is air, identically.
                //
                // Per-segment Beer's law happens at the top of the next
                // bounce iteration via the segment-absorption block.
                // This branch only flips state and re-aims the ray; the
                // previous nominal 1.5m entry estimate (and the b == 0
                // exit-side hack from b2690ac) are subsumed by the new
                // generic per-bounce attenuation.
                // #305 needs the stack AS IT WAS on the far side of this
                // interface: its sun shadow ray starts just OUTSIDE the
                // water and must not pick up the water absorption the
                // updated stack would apply along its whole length.
                uint medium_before_cross = medium;
                medium = (ndi_s < 0.0) ? ptMediumEnter(medium, MEDIUM_WATER)
                                       : ptMediumPop(medium);
                rd = normalize(refracted);
                ro = ptRayOrigin(hit_pt, -nf_s, cone_hit, rd);
                // --- #305: arm the water-leaving source -------------------
                //
                // We have just crossed INTO water from outside, which is
                // the case every in-scatter term in this shader skipped.
                // Everything the closed form needs is known here except
                // the column's length, so compute the amplitude and the
                // saturation rate now and spend them at the top of the
                // next iteration, where the trace supplies the length.
                //
                // THE CLOSED FORM. For a homogeneous column lit by a
                // collimated beam, the single-scattered radiance reaching
                // the interface from path length T, travelling back along
                // -rd, is
                //
                //   L = E_d(0-) beta(psi) [1 - exp(-k T)]
                //                       / (sigma_t (mu_s + mu_v))
                //   k = sigma_t (mu_s + mu_v) / mu_s
                //
                // with mu_s / mu_v the cosines of the REFRACTED sun and
                // view directions against the local vertical, beta the
                // volume scattering function at the in-water scattering
                // angle, and sigma_t = a + b the beam attenuation. It is
                // the integral the submerged march evaluates numerically,
                // done exactly; WaterOptics.h derives it and the test
                // pins the two against each other.
                //
                // WHY mu_a / mu_s ON THE IRRADIANCE. Refraction bends the
                // beam toward the vertical, so the same power crosses a
                // LARGER in-water cross-section: E_perp,w = E_perp,a
                // (1 - F) mu_a / mu_s. Equivalently the downwelling
                // irradiance on the horizontal obeys the textbook
                // E_d(0-) = E_d(0+) (1 - F), which is what this reduces
                // to. The n^2 of the radiance transform and the 1/n^2 of
                // the solid-angle compression do NOT simply cancel -- the
                // Jacobian is mu_a / (n^2 mu_s), not 1 / n^2 -- and the
                // submerged march below carried the cancelled version.
                // Both now use one factor, spelled out once each.
                if (ndi_s < 0.0 && water_scatter2.z > 0.0 &&
                    uint(sun_and_mode.w) >= 2u) {
                    float3 up_w    = localUp(hit_pt);
                    float3 sun_dir = sun_and_mode.xyz;
                    float  mu_a    = dot(up_w, sun_dir);
                    float  mu_v    = -dot(up_w, rd);   // rd points DOWN
                    if (mu_a > 0.0 && mu_v > 0.0) {
                        float3 into = refract(-sun_dir, up_w,
                                              1.0 / water_ior);
                        if (dot(into, into) > 1e-6) {
                            into = normalize(into);
                            float mu_s = -dot(up_w, into);
                            // The sun's own crossing of the interface,
                            // taken on the MACRO vertical because the
                            // stratification is in depth, not in wave
                            // slope. Same Schlick F0 as the split above.
                            float fr_sun = fresnelSchlick(mu_a, F0_water);
                            float3 sun_t = ocean_sun_trans;
                            if (sun_t.r < 0.0) {
                                sun_t = transmittance(
                                    ptRayOrigin(hit_pt, nf_s, cone_hit,
                                                sun_dir),
                                    sun_dir, 1e30, shutter_t01, cone_hit,
                                    medium_before_cross);
                            }
                            if (mu_s > 0.0 && any(sun_t > float3(0.0))) {
                                // E_d just below the surface, W/m^2.
                                float3 e_d = sunIrradianceAt(hit_pt, sun_dir,
                                                             mu_a)
                                           * sun_t * mu_a * (1.0 - fr_sun);
                                // The in-water scattering angle, in the
                                // same orientation the submerged march
                                // uses: cos psi = dot(view_prop, to_sun).
                                float  mu_sc = dot(rd, -into);
                                float3 s_mol = max(water_scatter.xyz,
                                                   float3(0.0));
                                float  s_par = max(water_scatter.w, 0.0);
                                // c_eff = a + b_b (Gordon 1973), the same
                                // law transmittance() and the submerged
                                // march use. See ptWaterEffAtten.
                                float3 sig_t = ptWaterEffAtten();
                                float  g_p   = clamp(water_scatter2.x,
                                                     -0.99, 0.99);
                                float  dpol  = clamp(water_scatter2.w,
                                                     0.0, 1.0);
                                float3 beta  =
                                      s_mol * ptPhaseRayleighDepol(mu_sc,
                                                                   dpol)
                                    + float3(s_par)
                                        * ptPhaseHenyeyGreenstein(mu_sc, g_p);
                                float  msmv  = mu_s + mu_v;
                                float3 den   = sig_t * msmv;
                                if (all(den > float3(0.0))) {
                                    // n^2: radiance leaving water for air
                                    // scales by 1/n^2. The (1 - F) of the
                                    // upward crossing is NOT here -- the
                                    // Fresnel roulette that put us on this
                                    // branch already is it.
                                    float n2 = water_ior * water_ior;
                                    wl_amp   = e_d * beta / (den * n2);
                                    wl_rate  = den / mu_s;
                                    wl_pending = any(wl_amp > float3(0.0));
                                    wl_primary = (b == 0u);
                                }
                            }
                        }
                    }
                }
                // --- end #305 ---------------------------------------------
                // Transmitted lobe: PERSIST skip_sky_on_miss /
                // prev_was_lambert_nee across the interface. Water is on
                // transmittance()'s transmissive-blocker list, so a
                // preceding Lambert env-NEE already accounted for the sky
                // arriving through this same water; the eventual sky-on-
                // miss down this chain must not be re-counted. See the
                // MAT_DIELECTRIC refract branch for the full rationale.
                // (Underwater misses hit the deep-water override above and
                // never reach the sky term, so this only matters once the
                // ray has refracted back out into air.)
            }
        }
#endif  // PT_WATER_ENABLED

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
        // Skip surface aerial perspective when the primary segment was
        // underwater (Phase 4, #134): the camera-to-surface segment was
        // already attenuated by per-segment Beer's law in the bounce
        // loop; multiplying by atmospheric transmittance would double-
        // dim AND mix air-tuned Mie/Rayleigh sigmas with what's actually
        // a water medium. The HDRI-haze additive at line ~4791 is also
        // sky-coloured so it's wrong underwater for the same reason.
        if (primary_hit && !primary_in_water) {
            float3 trans_atm_surf = atmosphericTransmittance(
                primary_ro, primary_rd, primary_t);
            radiance *= trans_atm_surf;
            // B10 SIGMA demod: the captured direct-sun terms are part of
            // `radiance`, so apply the same aerial-perspective attenuation
            // to keep the demod subtraction / re-modulation consistent.
            // No-op (zero) unless this pixel captured sun above.
            sample_sun_shadowed *= trans_atm_surf;
            sample_sun_direct   *= trans_atm_surf;
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
            // direction. skyColor(primary_ro, primary_rd, seed) samples
            // env_map for
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
                    skyColor(primary_ro, primary_rd, seed) * (float3(1.0) - trans_atm_surf));
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
        //
        // Smoke (#136) piggybacks on the cloud march -- the inner branch
        // at ~4378 enables the slab when smoke_active even with clouds
        // off, but the OUTER gate must also include smoke_active or the
        // whole block is skipped on a smoke-only scene (r_clouds 0 +
        // r_volumetric 0 + r_smoke_enabled 1). Without this OR-clause
        // smoke_emit plumes never render despite the inner-branch logic
        // being correct. Mirrors the (smoke_params.y > 0 && smoke_params.x > 0)
        // gate used inside the cloud-march body.
        bool smoke_active_outer = (smoke_params.y > 0.0) && (smoke_params.x > 0.0);
        // Fluid Phase 3 (#22): SPH particles bound -> the smoke-only
        // outer cloud-march gate must trigger so the splat list gets
        // marched against the slab. Mirror the gate semantics from
        // cloud_optical_depth.
        if (smoke_sph_params.x > 0.0) smoke_active_outer = true;
        // Skip atmospheric in-scatter / cloud / smoke march when the
        // primary segment was underwater (Phase 4, #134): all three
        // contributions assume an air medium with sky-coloured single
        // scatter, which doesn't apply when the primary ray travelled
        // through water before hitting (or missing) scene geometry.
        // A proper subsurface scattering integral lands in Phase 5; for
        // Phase 4 we just don't add air-only volumetric terms to an
        // underwater frame.
        if ((vol_params.x > 0.0 || clouds_p1.w > 0.0 || clouds_p3.w > 0.0
             || smoke_active_outer)
            && primary_t_recorded
            && !primary_in_water) {
            float  g         = vol_params.y;
            float  vol_int   = vol_params.z;
            int    N         = max(1, int(vol_params.w));
            float3 sun_dir   = sun_and_mode.xyz;
            // Planetary P0 (#254): solar elevation at the ray origin (the
            // camera). Deliberately hoisted, like sun_rad below -- the
            // march's per-sample points span at most 60 km, over which the
            // local-up direction turns by ~0.5 deg, and the hoist is what
            // keeps sun_rad a loop invariant for the B10 SIGMA demod.
            // The per-sample gates inside the march use planetAltitude()
            // for altitude, which IS per-point. Planar frame ->
            // sun_dir.y bit-for-bit.
            float  sun_elev  = elevationAt(primary_ro, sun_dir);
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
                if (sun_extra2.y > 0.0) {
                    // Physical: the same slant-path integral the NEE
                    // site uses, so the light the clouds are lit by and
                    // the light the ground is lit by cannot disagree.
                    sun_rad = kPtSolarIrradiance
                            * sunSlantTransmittance(primary_ro, sun_dir);
                } else {
                    // Legacy tuned curve, byte-identical to pre-#257.
                    // Note it uses 0.30 for the warmth ramp where the
                    // NEE site uses 0.18 and a different warm endpoint:
                    // two copies of one physical quantity that had
                    // already drifted apart, which is its own argument
                    // for computing it instead of authoring it.
                    float tau = 0.30;
                    float t_atm = exp(-tau * sun_am);
                    float warmth = saturate((sun_am - 1.0) * 0.30);
                    // #280: one documented scale, kPtLegacySkyScale, on
                    // the grey level -- this branch paints its own tint
                    // and multiplying a painted tint by the physical
                    // triple would tint twice.  Same treatment as
                    // sunIrradianceAt's legacy branch, which this is the
                    // drifted second copy of.
                    sun_rad = lerp(float3(1.00, 0.97, 0.92),
                                   float3(1.55, 0.40, 0.12), warmth)
                            * (80.0 * kPtLegacySkyScale * t_atm);
                }
            }
            // HG in-scatter cosine: incident light propagates as -sun_dir
            // (sun_dir points toward the sun), outgoing toward camera is
            // -primary_rd, so cos = dot(-sun_dir,-primary_rd) =
            // dot(primary_rd, sun_dir). Peaks the forward haze glow toward
            // the sun (dot(...,-sun_dir) had it on the anti-solar side).
            float cos_theta = dot(primary_rd, sun_dir);
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
            // Planetary P3 (#257): the medium comes from the body, not
            // from literals. This block used to carry its own copy of
            // the Rayleigh sigma triple and BOTH scale heights, so the
            // in-scatter march and atmosphericTransmittance could --
            // and did -- disagree about what the air was made of the
            // moment one was edited. One constructor now feeds both.
            PtAtmoBody atm_body = atmosphereBody();
            // The body's centre, for the per-sample multiple-scattering
            // lookup (#280).  Read out of the SAME pushed field the
            // atmosphere shell and the ground body read, which is #253's
            // rule: two copies of "where is the planet" is a horizon in
            // two places.
            float  planet_R_ms = sun_extra.w;
            float3 pc_v_ms = (planet_center_radius.w > 0.0)
                           ? planet_center_radius.xyz
                           : float3(0.0, -max(planet_R_ms, 6371008.8), 0.0);
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
            // --- #280: who owns the sky's molecular term -------------------
            //
            // procSky's palette is no longer painted -- the host integrates
            // the Rayleigh + ozone + multiple-scattering sky at three
            // anchor directions each frame and procSky interpolates them
            // (sky_cook_*).  So on a primary MISS in procedural mode the
            // molecular half of this march is now the SECOND estimator of
            // the same light, and before #280 that double count was the
            // sky's dominant term: r_rayleigh 30 made the march about 13x
            // the painted base it was supposed to be topping up.
            //
            // The aerosol half stays.  It is the sharp forward lobe around
            // the sun -- god rays, the sunset haze glow -- which a
            // three-anchor interpolation cannot represent and which the
            // cook therefore deliberately excludes.  The split is by
            // frequency content: smooth term cooked, peaked term marched,
            // neither counted twice.
            //
            // A primary HIT is untouched: nothing else integrates
            // camera-to-surface air, and that is where aerial perspective
            // comes from.
            if (uint(sun_and_mode.w) == 2u && !primary_hit) march_ray = false;
            // --- Planetary P6 (#260): the span, and who owns it ------------
            //
            // In the physical sky mode a primary MISS has already had its
            // entire column integrated by skyPhysical at the miss site, so
            // marching it again here would double-count it. A primary HIT
            // has not: nothing else integrates camera-to-surface air, and
            // this march is where aerial perspective comes from.
            //
            // For a hit, the 60 km cap has to go. It was a sensible bound
            // for a ground camera -- Rayleigh's 8 km scale height puts
            // almost everything in the bottom 25 km -- but from 400 km
            // looking down, the FIRST 60 km of the ray is vacuum and the
            // air is in the last 100. Capping there gives the surface
            // extinction with no in-scatter at all, so the planet reads as
            // a dark body with no haze over it: the same failure as the
            // black disc, one term further along.
            //
            // The replacement is not a bigger number. It is the shell clip
            // (ptRayShell against the body's own top_radius), which spends
            // the same N samples on the air wherever the camera is -- the
            // property that makes the in-scatter continuous across the
            // climb for exactly the reason skyPhysical's own march is.
            //
            // The legacy modes keep min(primary_t, 60000) bit-for-bit.
            uint sky_mode_v = uint(sun_and_mode.w);
            bool physical_sky_v = (sky_mode_v == 4u);
            float t_near_v = 0.0;
            float t_far_v  = min(primary_t, 60000.0);
            bool span_ok_v = true;
            if (physical_sky_v) {
                if (primary_hit) {
                    float planet_R_v = sun_extra.w;
                    float3 pc_v = (planet_center_radius.w > 0.0)
                                ? planet_center_radius.xyz
                                : float3(0.0, -planet_R_v, 0.0);
                    span_ok_v = (planet_R_v > 0.0)
                             && ptRayShell(primary_ro, primary_rd, pc_v, 0.0,
                                           ptAtmoEarth(planet_R_v).top_radius,
                                           primary_t, t_near_v, t_far_v);
                    // AND STOP THE COLUMN AT THE GROUND (#300 follow-up).
                    // skyPhysical has carried this clamp since #260 -- see
                    // "The body is opaque" next to its own ptRayShell call
                    // -- and this march did not, which made the two
                    // disagree about where the air ends for exactly the
                    // rays that hit the body.
                    //
                    // It bites whenever the primary hit is BELOW the
                    // atmosphere's datum, and with streamed terrain that is
                    // every pixel of the disc: the datum is the site's
                    // geocentric radius while P3's analytic backstop is
                    // WGS-84 b - 10 935 - 65 m, 27.7 km lower at the 27.99 N
                    // site. Those 27.7 km were marched with the altitude
                    // clamped to 0, i.e. as a slab of SEA-LEVEL air, and
                    // being in-scatter it ADDED light rather than removing
                    // it. Measured at nadir, sub-solar: the slab contributes
                    // sigma_s(0) * 27.7 km * P_R(-1) * E = (7.4, 21.6, 54.4)
                    // W/m^2/sr against a true column in-scatter of
                    // (2.1, 6.2, 15.7) -- four times the real atmosphere,
                    // which is what washed the land cover off the disc.
                    //
                    // Same defect as the one ptAtmoOpticalDepth carries on
                    // the extinction side, and the same fix: the medium has
                    // a floor.
                    //
                    // WHY NO IMAGE GOLDEN MOVES, AND WHY THAT IS THE
                    // POINT. Every fixture with r_sky_mode physical has
                    // r_planet_terrain 0, and without terrain the analytic
                    // body sits AT the datum (Engine.cpp only lowers it to
                    // the backstop when terrain is on), so this clamp is a
                    // no-op on all nine of them -- verified: the whole
                    // matrix is bit-stable across this change. That is
                    // exactly the gap the defect lived in. Until a cell
                    // combines the two, the guard is the source pin in
                    // tests/pt_atmosphere_test.cpp, not a picture.
                    float g0_v, g1_v;
                    if (span_ok_v
                        && ptSphereRoots(primary_ro, primary_rd, pc_v,
                                         planet_R_v, g0_v, g1_v)
                        && g0_v > 0.0) {
                        t_far_v = min(t_far_v, g0_v);
                        span_ok_v = (t_far_v > t_near_v);
                    }
                } else {
                    span_ok_v = false;   // skyPhysical already did this ray
                }
            }
            if ((march_mie || march_ray) && span_ok_v) {
                float  t_far   = t_far_v;
                float  dt      = (t_far - t_near_v) / float(N);
                // Stratified jitter IN [0, 1) of a cell -- combined with
                // the plain (vi + jit) offset below this covers
                // [0, t_far) exactly. The old mid-cell + full-cell-
                // jitter form sampled [0.5dt, t_far + 0.5dt): the first
                // half step of haze in front of the camera (~1.9 km at
                // the 60 km / 16-sample default!) was NEVER sampled --
                // a bias no amount of temporal accumulation converges
                // away, and ground fog is densest exactly there.
                float  jitter  = randf(seed);
                float3 trans_eye = float3(1.0);
                float3 v_color = float3(0.0);
                // Aerosol phase. The legacy modes keep plain
                // Henyey-Greenstein, bit-for-bit. The physical mode uses
                // Cornette-Shanks -- the same helper skyPhysical uses --
                // because the two marches meet at the horizon: one pixel
                // hits the body and gets its in-scatter from here, the
                // pixel above it misses and gets it from skyPhysical, and
                // if they disagreed about the phase function there would
                // be a seam along the limb. The two differ by up to 2x
                // (CS/HG = 3(1 + mu^2) / (2(2 + g^2)), which is 0.60 at
                // mu = 0 and 1.20 at mu = 1 for g = 0.7), so the seam
                // would be plainly visible.
                float phase_mie_v = physical_sky_v
                                  ? ptPhaseCornetteShanks(cos_theta, g)
                                  : phase;
                // Identical math to the (3/(16 pi))(1 + mu^2) literal this
                // replaces; routed through the shared helper so the sky
                // modes and the haze march cannot drift apart.
                float phase_ray_v = ptPhaseRayleigh(cos_theta);
                // Issue #256: a march sample sits in open air, so its only
                // scale is the primary cone -- exactly the footprint the
                // pixel is integrating over at that distance.
                float cone_spread_v = (2.0 * pos_fovtan.w) / float(max(dim.y, 1u));
                for (int vi = 0; vi < N; ++vi) {
                    // `t_near_v + x` with t_near_v == 0.0 is x bit-for-bit,
                    // which is what keeps the legacy modes unchanged.
                    float t = t_near_v + (float(vi) + jitter) * dt;
                    if (t >= t_far) break;
                    float3 sp = primary_ro + primary_rd * t;
                    PtRayCone cone_v = ptConeMake(cone_spread_v * t, cone_spread_v);
                    // Per-point altitude above the surface (#254). Planar
                    // frame -> sp.y bit-for-bit.
                    float h = max(planetAltitude(sp), 0.0);
                    // Three species (#257). sigma_s_* are what SCATTERS
                    // into the eye; sigma_t is what ATTENUATES along it,
                    // and the two are no longer the same quantity -- Mie
                    // absorption and ozone are extinction with no
                    // scattering partner. Before this the march used
                    // scattering as extinction, so the haze column
                    // transmitted more light than it physically could.
                    float3 sigma_s_ray, sigma_s_mie3, sigma_total;
                    ptAtmoCoefficients(atm_body, h,
                                       sigma_s_ray, sigma_s_mie3,
                                       sigma_total);
                    float  sigma_mie = march_mie ? sigma_s_mie3.g : 0.0;
                    float3 sigma_ray = march_ray ? sigma_s_ray : float3(0.0);
                    // Sunlight AT THIS SAMPLE.
                    //
                    // The legacy path uses the hoisted camera-elevation
                    // value and gates the whole march on the sun being up
                    // at the CAMERA. That is why twilight has no
                    // in-scatter at all today: at -4 degrees the gate is
                    // false and the march contributes nothing, however
                    // much sunlit air is overhead.
                    //
                    // The physical path evaluates the slant-path
                    // transmittance at the sample, and needs no elevation
                    // gate -- a sample whose sun ray runs into the body
                    // gets exactly zero, which is what "in shadow" means.
                    // That per-sample geometry IS twilight: the observer
                    // is inside the planet's shadow while the air 20 km
                    // above is still lit. It is also the only arrangement
                    // in which ozone can colour anything, because it is
                    // the only one where the light reaching the eye has
                    // crossed the layer.
                    float3 sun_at_sample;
                    if (sun_extra2.y > 0.0) {
                        sun_at_sample = kPtSolarIrradiance
                                      * sunSlantTransmittance(sp, sun_dir);
                    } else {
                        sun_at_sample = (sun_elev > 0.0) ? sun_rad
                                                         : float3(0.0);
                    }
                    if (any(sun_at_sample > float3(0.0))) {
                        float3 sun_jit = sampleSunDisc(sun_dir, seed);
                        float3 trans_sun = transmittance(sp, sun_jit, 1e30,
                                                         shutter_t01, cone_v);
                        if (any(trans_sun > float3(0.0))) {
                            float3 in_scatter =
                                  float3(sigma_mie) * phase_mie_v
                                + sigma_ray         * phase_ray_v;
                            v_color += sun_at_sample * trans_sun * in_scatter
                                     * dt * trans_eye;
                        }
                    }
                    // Second and higher orders (#280), the same table the
                    // sky march reads.  Isotropic, so no phase and no sun
                    // visibility gate: aerial perspective at twilight is
                    // mostly this term, and gating it on the sun being
                    // visible FROM THE SAMPLE is what left the pre-#280
                    // twilight haze black.  Off in the legacy in-scatter
                    // A/B for the same reason the legacy sun curve is: the
                    // two halves of an A/B have to differ in one thing.
                    if (sun_extra2.y > 0.0) {
                        float3 up_s = normalize(sp - pc_v_ms);
                        float3 ms_s = ptAtmoMultiScatter(
                            atm_body, length(sp - pc_v_ms),
                            dot(up_s, sun_dir));
                        v_color += (sigma_ray + float3(sigma_mie)) * ms_s
                                 * dt * trans_eye;
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
            //
            // Phase 4 (#100) -- heterogeneous volumetric raymarching. The
            // legacy march was uniform-stride single-scatter with a haze-
            // shared HG g; Phase 4 adds:
            //
            //   * Density scale -- vol_density_scale (push.vol_density_scale)
            //     multiplies the cloud sigma_t inside the march so users
            //     can dial volume thickness without changing the preset
            //     peak density. Carries through to shadow rays via
            //     cloud_optical_depth_stratified, so the shadow stays
            //     consistent with the body.
            //   * Cloud-specific HG g -- vol_phase_g_cloud. Decoupled from
            //     vol_params.y (atmospheric Mie haze g). Real cumulus is
            //     forward-scattering at g~0.8; without this cvar the
            //     cloud march was forced to share the haze's g~0.7 for
            //     both god-ray halo and silver lining, which under-fed
            //     the cumulus forward lobe.
            //   * Multi-scatter octaves (Wrenninge-style) -- approximates
            //     the diffuse fill that real cumulus gets from light
            //     bouncing several times inside the volume. Without
            //     this, deep cloud cores read as pitch-black under sun
            //     self-shadow; with N octaves, each octave reduces the
            //     extinction coefficient by 2^-i so the deeper octaves
            //     act as cheap proxies for multi-scattered light.
            //   * Stratified cloud-shadow rays via inline
            //     cloud_optical_depth_stratified * atmosphericTransmittance
            //     (instead of a traceScene call). Sharper god rays through
            //     thin cloud gaps. Skips the expensive traceScene call for
            //     the open-sky leg above the cloud (the cloud + atmosphere
            //     is the only occluder by construction), which is what
            //     keeps the per-sample NEE cost under budget when crossing
            //     12+ sub-samples stratified along the shadow ray.
            //   * Adaptive step size -- when the current sample finds
            //     near-zero density we increment t by the same dtc
            //     anyway (no work skipped); the win is that the per-
            //     step ALU is tiny in empty space and the dense steps
            //     get the full multi-scatter + dual-NEE pass. This
            //     keeps the GPU's wavefront divergence-friendly because
            //     every lane in a warp executes the same loop iteration
            //     count -- only the early-out work inside the lane
            //     changes.
            // Fluid Phase 1 (#136): the cloud march also drives smoke
            // emitter rendering -- additive Gaussian density blobs
            // composed into the same Beer-Lambert / NEE pipeline as
            // the cloud field. When clouds are off but smoke is on we
            // still need to march, just with the slab clipped to span
            // the smoke layer rather than the (unset) cloud band.
            // Smoke is "active" when either the procedural Phase 2 emitters
            // are bound (smoke_params.y gate + count) OR the Phase 3 SPH
            // particles are bound (smoke_sph_params.x > 0). The host's mode
            // selection in r_smoke_mode collapses one of the two to zero so
            // only one path contributes per frame; the OR keeps the gate
            // logic generic.
            bool smoke_active = (smoke_params.y > 0.0) && (smoke_params.x > 0.0);
            bool sph_active   = (smoke_sph_params.x > 0.0);
            smoke_active = smoke_active || sph_active;
            // Wave 7 (#24): when r_clouds_mode == procedural_raymarched
            // the inline cloud-march below is replaced by a dedicated
            // compute pre-pass (CloudsRaymarch.slang). The pre-pass
            // owns clouds_color_tex + cloud_trans_tex writes; PathTrace
            // skips the cloud-density sampling entirely so the two
            // modes don't double-count cloud in-scatter / cloud
            // transmittance. The smoke-only branch below (when
            // clouds_p1.w <= 0 && smoke_active) STILL runs since the
            // raymarched cloud pre-pass doesn't handle smoke.
            //
            // Pathtraced mode (clouds_runtime.x == 0) preserves the
            // existing inline march bit-for-bit -- this gate adds zero
            // ALU on that path because the Slang optimizer constant-
            // folds the comparison when the same value is supplied for
            // every dispatch (default cvar = 0). Even on a runtime
            // value, the cost is one uniform branch.
            bool clouds_raymarched_mode = (clouds_runtime.x != 0u);
            bool inline_cloud_march_active =
                (clouds_p1.w > 0.0 && !clouds_raymarched_mode) || smoke_active;
            if (inline_cloud_march_active) {
                float base_y = clouds_p1.x;
                float top_y  = clouds_p1.y;
                // When the cloud field is off, drop the slab down to
                // ground level so chimney / campfire-altitude smoke
                // (typical y in [0, 50]) actually gets marched. Cap
                // at 2 km so the per-pixel sample budget doesn't get
                // wasted in pure-sky pixels.
                if (clouds_p1.w <= 0.0) {
                    base_y = 0.0;
                    top_y  = 2000.0;
                } else if (smoke_active) {
                    // Clouds ON but smoke is also active: the cloud band
                    // (base_y..top_y, ~250-800 m) doesn't cover ground-level
                    // chimney / campfire smoke (y in [0,50]). Without
                    // extending the slab down to the ground the plume is
                    // never sampled -- invisible and shadowless -- whenever
                    // the cloud layer happens to be enabled. Keep the cloud
                    // top; just lower the floor so the march spans both.
                    base_y = min(base_y, 0.0);
                }
                float t_in, t_out;
                if (!cloudLayerInterval(primary_ro, primary_rd,
                                        base_y, top_y, primary_t,
                                        t_in, t_out)) {
                    t_in = 1.0; t_out = 0.0;   // miss
                }
                // Cap the cloud march span at 30km. For low-elevation
                // rays the layer can span hundreds of km; uniform
                // sampling that range produces samples spaced wider
                // than cloud features and looks worse than just
                // marching the closer 30km. Beyond 30km the sky
                // gradient takes over visually.
                // Keep the NEAR 30 km window [t_in, t_in+30km], not the far
                // one: t_in is the slab entry nearest the ray origin, so a
                // plume/cloud a few tens of metres ahead must stay inside
                // the marched span. The old t_in = t_out - 30000 kept the
                // farthest 30 km and silently dropped the near field (its
                // own comment says it marches "the closer 30km").
                //
                // PLANAR FRAME ONLY (planetary P3, #257). On a shell the
                // cap would defeat the whole point of the shell: a
                // ground camera's horizon-grazing chord through a
                // 200-500 m layer is 50-80 km long, so truncating at
                // 30 km ends the cloud deck in mid-air well before the
                // horizon -- reintroducing #51's founding symptom by a
                // different route. The shell interval is already finite
                // and already terminates at the horizon, so the cap has
                // nothing left to protect against; the sample budget is
                // held instead by the transmittance early-out further
                // down (`if (trans_eye_c < 0.003) break;`), which is
                // what a march wants anyway and is altitude-agnostic.
                if (planet_center_radius.w <= 0.0
                    && t_out - t_in > 30000.0) {
                    t_out = t_in + 30000.0;
                }
                if (t_out > t_in) {
                    // Wave-6: when smoke is active (especially smoke-only
                    // scenes where the cloud field is off), the per-pixel
                    // sample budget needs to be high enough that small
                    // (~1-3m radius) smoke columns get more than 1 sample.
                    // The legacy max(N, 12) was tuned for big diffuse
                    // clouds where 12 samples across a 30km span is
                    // plenty; for a 100m smoke march span 12 samples gives
                    // 8m step, larger than the per-puff radius so the
                    // shape reads as a single solid blob.
                    //
                    // Heuristic: when smoke is active and the march span
                    // is short (< 200m, i.e. ground-level chimney /
                    // campfire range, not horizon-spanning cloud bank),
                    // bump Nc so dtc lands ~0.3-1m -- finer than the
                    // smallest typical smoke puff radius. Bounded at 48
                    // so the per-pixel cost stays under budget even when
                    // r_volumetric_samples is the default 16.
                    int   Nc = max(N, 12);    // user slider can ask for more
                    if (smoke_active && (t_out - t_in) < 200.0) {
                        int target = int((t_out - t_in) * 1.0 + 0.5);  // ~1 sample/m
                        Nc = clamp(target, Nc, 48);
                    }
                    float dtc = (t_out - t_in) / float(Nc);
                    // Stratified: (vi + jitc) * dtc covers the slab
                    // exactly (see the Mie/Rayleigh march above for the
                    // half-cell-bias rationale).
                    float jitc = randf(seed);
                    float trans_eye_c = 1.0;
                    float3 v_color_c  = float3(0.0);
                    // Phase 4 cvar pull. vol_density_scale clamps to a
                    // safe non-zero floor (a literal 0 would make the
                    // cloud march transmit unconditionally; explicit
                    // "no cloud" is what clouds_p1.w == 0 means).
                    // vol_phase_g_cloud is bounded inside HG's stable
                    // range; outside +/-0.95 the (1+g^2-2gcos) term in
                    // the denominator can pinch to zero on grazing
                    // directions and produce per-sample firefly spikes.
                    // multiscatter_oct clamps to 0..4: 0 collapses the
                    // multi-scatter sum to a no-op (legacy behaviour);
                    // 4 is the deepest we trust before each octave's
                    // contribution falls below the ACES knee.
                    float vol_dscale = max(vol_density_scale, 1e-3);
                    float g_cloud    = clamp(vol_phase_g_cloud, -0.95, 0.95);
                    int   ms_oct     = clamp(int(vol_multiscatter_oct + 0.5),
                                              0, 4);
                    float g_cloud2   = g_cloud * g_cloud;
                    // (Per-light HG phase is recomputed inside each NEE
                    // branch below because cos_theta is per-light;
                    // pre-computing a primary-ray phase value here would
                    // be dead in the cloud march's hot path.)
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
                        float t = t_in + (float(vi) + jitc) * dtc;
                        if (t > t_out) break;
                        float3 sp = primary_ro + primary_rd * t;
                        // Phase 4: heterogeneous density (procedural fbm
                        // via cloud_density_at) multiplied by the user
                        // density-scale knob. Sigma_t varies through the
                        // volume at every step -- this is what makes the
                        // march "heterogeneous" vs the old homogeneous
                        // single-sigma haze path. The cloud field's own
                        // fbm + height profile already provides the
                        // spatial variation; vol_dscale is a global
                        // multiplier on the per-metre extinction.
                        // #117 added clouds_p4 (curl-noise params).
                        // #100 added vol_density_scale multiplier + empty-cell
                        // skip optimization. Both layered.
                        // Wave 7 (#24): when r_clouds_mode == raymarched
                        // the cloud field is sampled entirely by the
                        // CloudsRaymarch pre-pass; the inline march here
                        // skips cloud_density_at to avoid double-counting
                        // cloud in-scatter when the composite step layers
                        // the pre-pass result on top of the PT output.
                        // Smoke-only paths still get the normal behaviour
                        // (smoke is NOT handled by the raymarch pre-pass).
                        float cloud_dens = clouds_raymarched_mode
                            ? 0.0
                            : cloud_density_at(sp, clouds_p1,
                                               clouds_p2, clouds_p3,
                                               clouds_p4, 6,
                                               planet_center_radius)
                              * vol_density_scale;
                        // Fluid Phase 1 (#136): add density-injection
                        // smoke contributions on top of the procedural
                        // cloud field. Smoke shares the cloud march's
                        // NEE, multi-scatter, and aerial-perspective
                        // pipeline -- it just shows up as "extra
                        // density" at this sample and gets shaded
                        // identically to a cloud cell. The tinted
                        // variant also returns the density-weighted
                        // average per-emitter tint (v2.xyz on the
                        // smoke SSBO) so the NEE radiance contributions
                        // below can multiply by lerp(white, smoke_tint,
                        // smoke_fraction) -- this is the missing piece
                        // that lets per-plume colour actually reach the
                        // framebuffer instead of being a documented
                        // no-op.
                        float3 smoke_tint = float3(1.0);
                        float  smoke_dens = smoke_density_add_tinted(sp, smoke_tint);
                        // Fluid Phase 3 (#22): the SPH particle splat
                        // list contributes a parallel density + tint
                        // here. The two tinted contributions blend by
                        // their respective density weights so an SPH-
                        // mode plume picks up its age gradient (warm
                        // fresh -> cool old) the same way the Phase 2
                        // procedural plume does. Returns 0 / white
                        // when no SPH particles are bound.
                        float3 sph_tint   = float3(1.0);
                        float  sph_dens   = smoke_sph_density_add_tinted(sp, sph_tint);
                        // Density-weighted blend of the two tints so a
                        // plume that's a mix of both contributions sees
                        // its NEE radiance painted by the heavier one.
                        float combined_smoke = smoke_dens + sph_dens;
                        if (combined_smoke > 0.0) {
                            smoke_tint = (smoke_dens * smoke_tint + sph_dens * sph_tint)
                                          / combined_smoke;
                        }
                        smoke_dens = combined_smoke;
                        float  dens       = cloud_dens + smoke_dens;
                        float  smoke_frac = (dens > 0.0) ? (smoke_dens / dens) : 0.0;
                        // Effective per-light tint at this sample: white
                        // when there's no smoke contribution (clear-cloud
                        // sample), tends to the smoke tint as the smoke
                        // density dominates the local extinction.
                        float3 light_tint = lerp(float3(1.0), smoke_tint, smoke_frac);
                        if (dens <= 0.0) {
                            // Empty cell -- transmittance does not change,
                            // no light scatters in. Skip per-light NEE for
                            // this step; the cost asymmetry collapses to
                            // "one density eval" in clear sky vs full work
                            // in cloud.
                            continue;
                        }
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
                        // --- Multi-scatter Wrenninge octaves ----------------
                        // Approximates the diffuse light that has bounced
                        // multiple times inside the cloud and now exits
                        // toward the camera. Cheap proxy: re-evaluate the
                        // direct-NEE contribution under reduced extinction
                        // (each octave halves the effective sigma_t) so
                        // light penetrates "further into the cloud" each
                        // octave -- this is the deep-core fill that pure
                        // single-scatter misses. Phase function flattens
                        // toward isotropic each octave because diffuse
                        // multi-scatter loses anisotropy with bounce count.
                        //   ms_weight[i] = a^i           absorption decay
                        //                  b^i (phase)   anisotropy decay
                        // Coefficients (a, b, c) follow Wrenninge 2013
                        // ("Oz: The Great and Volumetric"); a=0.5, b=0.5,
                        // c=0.5 reads cleanly across overcast and cumulus
                        // without per-preset re-tuning.
                        // 0 octaves => skip entirely (legacy behaviour);
                        // >=1 octave => sum starts at i=0 single-scatter,
                        // i=1..ms_oct are the multi-scatter fakes.
                        if (hdri_lit) {
                            // HDRI extracted-lights NEE: stochastic
                            // single-pick weighted by cluster luminance.
                            // Phase 4: compose the shadow visibility inline
                            // as cloud_optical_depth_stratified *
                            // atmosphericTransmittance so the cloud's own
                            // stratified Beer-Lambert produces sharp god
                            // rays through holes between cumulus puffs.
                            if (hdri_lights_count > 0u) {
                                HdriLightSample hl = sampleHdriLight(seed);
                                if (hl.pdf > 0.0) {
                                    float c_t_h = dot(primary_rd, hl.dir);
                                    float t_s_cloud = exp(-cloud_optical_depth_stratified(
                                                            sp, hl.dir, 1e30, seed));
                                    float3 t_s_atm = atmosphericTransmittance(
                                                            sp, hl.dir, 60000.0);
                                    float3 t_s = t_s_atm * float3(t_s_cloud);
                                    if (any(t_s > float3(0.0))) {
                                        // Sum direct + multi-scatter octaves.
                                        // Octave 0 is the single-scatter
                                        // direct contribution. Octave i>0
                                        // reuses the same shadow visibility
                                        // (t_s) with reduced phase
                                        // anisotropy (b^i toward isotropic)
                                        // and reduced effective extinction
                                        // -- the per-octave step_w_i applies
                                        // a^i so cloud cores still get fill
                                        // light even where direct sun
                                        // transmittance is ~0.
                                        float3 ms_acc = float3(0.0);
                                        float a_pow = 1.0;
                                        float b_pow = 1.0;
                                        for (int oi = 0; oi <= ms_oct; ++oi) {
                                            float g_o = g_cloud * b_pow;
                                            float g_o2 = g_o * g_o;
                                            float ph_o = (1.0 - g_o2) / (4.0 * 3.14159265358979
                                                * pow(max(1.0 + g_o2 - 2.0 * g_o * c_t_h, 1e-4), 1.5));
                                            float step_w_o = dens * a_pow * dtc * trans_eye_c;
                                            ms_acc += hl.color * t_s * ph_o * step_w_o;
                                            a_pow *= 0.5;
                                            b_pow *= 0.5;
                                        }
                                        v_color_c += clampIndirect(
                                            ms_acc * trans_atm * light_tint / hl.pdf);
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
                                    float c_t_e = dot(primary_rd, env.dir);
                                    float t_e_cloud = exp(-cloud_optical_depth_stratified(
                                                            sp, env.dir, 1e30, seed));
                                    float3 t_e_atm = atmosphericTransmittance(
                                                            sp, env.dir, 60000.0);
                                    float3 t_e = t_e_atm * float3(t_e_cloud);
                                    if (any(t_e > float3(0.0))) {
                                        float phase_e = (1.0 - g_cloud2)
                                            / (4.0 * 3.14159265358979
                                                * pow(max(1.0 + g_cloud2
                                                            - 2.0 * g_cloud * c_t_e, 1e-4), 1.5));
                                        v_color_c += clampIndirect(
                                            env.color * t_e * phase_e * step_w * trans_atm
                                            * light_tint / env.pdf);
                                    }
                                }
                            }
                        } else {
                            // Direct sun (single-scatter NEE). Altitude-
                            // boosted sun_rad: high clouds (cirrus at 8km)
                            // sit ABOVE most of the absorbing atmosphere
                            // and catch brighter / less-reddened light
                            // than the ground does. Phase 4: visibility
                            // composed inline (cloud_optical_depth_stratified
                            // * atmosphericTransmittance) for the sharp
                            // stratified cloud OD that god rays need to
                            // silhouette through cloud gaps.
                            if (sun_elev > 0.0) {
                                // Per-point altitude above the surface
                                // (#254). Planar frame -> sp.y bit-for-bit.
                                float h_alt   = max(planetAltitude(sp), 0.0);
                                // Reference height for the high-cloud sun
                                // boost is the Rayleigh scale height, which
                                // now comes from the body rather than from a
                                // literal that happened to equal it (#257).
                                float h_norm  = saturate(h_alt / atm_body.rayleigh_scale_h);
                                float boost   = exp(0.30 * sun_am * h_norm);
                                float3 sun_at = sun_rad * boost;
                                float3 sun_jit = sampleSunDisc(sun_dir, seed);
                                float c_t_sun = dot(primary_rd, sun_jit);
                                // Cloud + atmospheric leg for the shadow.
                                // Skips the BVH walk: by construction
                                // the only occluder along this leg is
                                // the cloud itself (the sun direction
                                // points to open sky above the cloud
                                // layer's top), so cloud_optical_depth_
                                // stratified + atmosphericTransmittance
                                // is the full physical visibility.
                                float t_s_cloud = exp(-cloud_optical_depth_stratified(
                                                        sp, sun_jit, 1e30, seed));
                                float3 t_s_atm = atmosphericTransmittance(
                                                        sp, sun_jit, 60000.0);
                                float3 trans_sun = t_s_atm * float3(t_s_cloud);
                                if (any(trans_sun > float3(0.0))) {
                                    // Multi-scatter octave sum (see HDRI
                                    // branch above for the rationale).
                                    // Direct + multi-scatter share the
                                    // SAME shadow visibility -- it's the
                                    // SAME sun ray; only the phase /
                                    // extinction weighting per octave
                                    // changes. Without the octaves the
                                    // cumulus core under self-shadow
                                    // reads pitch-black against the
                                    // brightly-lit cloud edges, which
                                    // looks like a clipping bug rather
                                    // than physics.
                                    float3 ms_acc = float3(0.0);
                                    float a_pow = 1.0;
                                    float b_pow = 1.0;
                                    for (int oi = 0; oi <= ms_oct; ++oi) {
                                        float g_o  = g_cloud * b_pow;
                                        float g_o2 = g_o * g_o;
                                        float ph_o = (1.0 - g_o2) / (4.0 * 3.14159265358979
                                            * pow(max(1.0 + g_o2 - 2.0 * g_o * c_t_sun, 1e-4), 1.5));
                                        float step_w_o = dens * a_pow * dtc * trans_eye_c;
                                        ms_acc += sun_at * trans_sun * ph_o * step_w_o;
                                        a_pow *= 0.5;
                                        b_pow *= 0.5;
                                    }
                                    v_color_c += clampIndirect(ms_acc * trans_atm * light_tint);
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
                                    // Same sharp cloud-aware visibility
                                    // as the sun branch; same rationale
                                    // (open sky above the cloud layer,
                                    // no other occluders).
                                    float t_m_cloud = exp(-cloud_optical_depth_stratified(
                                                            sp, moon_jit, 1e30, seed));
                                    float3 t_m_atm = atmosphericTransmittance(
                                                            sp, moon_jit, 60000.0);
                                    float3 trans_moon = t_m_atm * float3(t_m_cloud);
                                    if (any(trans_moon > float3(0.0))) {
                                        float c_t_m = dot(primary_rd, moon_dir_phase.xyz);
                                        float phase_m = (1.0 - g_cloud2)
                                            / (4.0 * 3.14159265358979
                                                * pow(max(1.0 + g_cloud2
                                                            - 2.0 * g_cloud * c_t_m, 1e-4), 1.5));
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
                                            moon_rad * trans_moon * phase_m * step_w * trans_atm * light_tint);
                                    }
                                }
                            }
                            // Diffuse skylight (omnidirectional, no phase
                            // function).
                            v_color_c += clampIndirect(sky_ambient * step_w * trans_atm * light_tint);
                        }
                        trans_eye_c *= exp(-dens * dtc);
                        // Early termination (Beer-Lambert front-to-back): once the
                        // accumulated eye transmittance is near zero the cloud is
                        // opaque along this ray, so every remaining march sample
                        // contributes step_w = dens*dtc*trans_eye_c ~ 0 while still
                        // paying a full multi-octave cloud_density_at + per-light
                        // shadow transmittance() march (and now the multi-scatter
                        // octave loop above). Bailing here skips the often-majority
                        // back-half samples on thick cloud pixels. Output changes by
                        // < ~1e-3 radiance -- under the golden tolerance and verified
                        // a no-op on the committed cloud goldens. Same SPIR-V / MSL
                        // codegen so Metal and Vulkan bail identically (no new
                        // cross-backend divergence). Threshold 0.003 ~= 99.7% opaque.
                        if (trans_eye_c < 0.003) break;
                    }
                    // Beer-Lambert background attenuation, opt-in via
                    // r_cloud_attenuate_background (piggybacked on
                    // smoke_params.w by Engine.cpp). When ON, the
                    // existing radiance accumulated for this primary
                    // ray (sun NEE + sky NEE + indirect) gets
                    // multiplied by the running transmittance product
                    // through the cloud volume, so a dense cloud /
                    // smoke plume actually OCCLUDES bright sky behind
                    // it. Real volumetric media work this way (Beer-
                    // Lambert + in-scatter together); the in-scatter
                    // half was already done by the v_color_c
                    // accumulation above, this is the missing half.
                    // Default OFF until all Wave-1 cloud goldens are
                    // re-rendered (#179).
                    if (smoke_params.w > 0.5) {
                        radiance *= trans_eye_c;
                        // B10 SIGMA demod: mirror the Beer-Lambert cloud
                        // attenuation onto the captured direct-sun terms so
                        // they track `radiance` (no-op unless sun captured).
                        sample_sun_shadowed *= trans_eye_c;
                        sample_sun_direct   *= trans_eye_c;
                    }
                    radiance += v_color_c * vol_int;
                    // G-buffer write source for cloud_trans_tex. Only
                    // sample 0's value is captured so the texture's
                    // value is deterministic when the dispatch finishes
                    // -- with spp>1 the cloud field is smooth enough
                    // that any single sample is a faithful proxy.
                    if (s == 0u) {
                        frame_cloud_trans = trans_eye_c;
                    }
                }
            }
        }

#if PT_WATER_ENABLED
        // --- Planetary P7 (#261): the underwater in-scatter march ----------
        //
        // WHAT WAS MISSING. All three air marches above are gated off by
        // !primary_in_water and NOTHING replaced them, so a submerged frame
        // was pure exponential darkening: absorption without scattering, no
        // colour build-up with distance, no shafts. That is the difference
        // between water and tinted glass, and it is why the sea floor and
        // the open water read as the same material at different depths.
        //
        // THIS IS THE SAME MARCH, ONE MEDIUM OVER. Deliberately the same
        // shape as the Mie/Rayleigh march above -- stratified samples with
        // one jitter, per-sample sun NEE through transmittance(), running
        // eye transmittance -- so the two can be read against each other
        // and neither can quietly acquire a different sampling bias. Three
        // things differ, and all three are physics:
        //
        //  1. TWO SPECIES, TWO PHASE FUNCTIONS. Molecular scattering is
        //     Rayleigh-like and near-symmetric; particulate scattering is
        //     violently forward (Petzold g = 0.924). #261 proposes a single
        //     HG lobe at g = 0.92 for both, which would make PURE water
        //     forward-scattering -- and pure water's near-isotropic blue
        //     scatter is precisely why a clear ocean is blue rather than
        //     merely dark. See src/physics/WaterOptics.h.
        //
        //  2. THE SUN IS NOT WHERE IT IS IN AIR. Sunlight entering water
        //     bends TOWARD the vertical, so from below the sun sits at
        //     asin(sin(theta_air) / n) from the zenith, never further than
        //     the critical angle. NEE toward the air direction would trace
        //     a ray that exits at the wrong angle entirely -- and beyond
        //     48.6 degrees would total-internal-reflect and return black
        //     for a sun that is plainly visible. Refracting the sun into
        //     the water first is what makes Snell's window SHADE correctly
        //     rather than merely be in the right place, which is all P5's
        //     geometric fix could do.
        //
        //     The radiometry works out with no extra factor. Radiance
        //     across the interface scales as L_w = L_a (1 - F) n^2 while
        //     the sun's solid angle compresses as Omega_w = Omega_a / n^2,
        //     so the product -- the irradiance, which is what the
        //     in-scatter integral wants -- is L_a Omega_a (1 - F). The n^2
        //     cancels exactly. transmittance() already applies the (1 - F)
        //     when the shadow ray crosses MAT_WATER, so the estimator is
        //     just sunRadianceAt * transmittance, unchanged from the air
        //     march. (sampleSunDisc still jitters over the AIR angular
        //     radius rather than the compressed one: it spreads the same
        //     energy over a cone 1.33x too wide, i.e. 0.53 deg instead of
        //     0.40 deg. Stated, not hidden.)
        //
        //  3. THE SPAN IS DERIVED, NOT CAPPED AT A ROUND NUMBER. Marching
        //     past the depth where the least-attenuated channel transmits
        //     under one 8-bit level spends samples on black. ln(255) /
        //     min(sigma_t) is 401 m in pure seawater and 185 m at #261's
        //     coastal c_B = 0.03 -- the engine computes it from whatever
        //     coefficients are actually in force, so it is a consequence of
        //     the water rather than a constant about it.
        //
        // GOD RAYS COME FREE AND CAUSTICS DO NOT. The per-sample shadow ray
        // is occluded by real geometry, so a submerged camera looking up
        // past a hull or a rock gets shafts out of this loop with no extra
        // term. Wave-DAPPLED shafts do not: transmittance() resolves the
        // ocean as its analytic shell (traceScene's MAT_WATER hit), not as
        // the displaced FFT surface, so the interface a shadow ray crosses
        // is smooth. And true caustics are a specular-diffuse-specular
        // path that neither BSDF sampling nor NEE can find at all -- that
        // is #86's MNEE, and #261 scopes it out explicitly.
        if (primary_in_water && primary_t_recorded
            && water_scatter2.z > 0.0) {
            float3 sigma_s_mol = max(water_scatter.xyz, float3(0.0));
            float  sigma_s_par = max(water_scatter.w, 0.0);
            float3 sigma_s     = sigma_s_mol + float3(sigma_s_par);
            // Extinction is absorption plus BOTH scattering species. The
            // air march learned this the hard way (#257): using scattering
            // as extinction lets the column transmit more light than it
            // physically can.
            // #305: the attenuation of the in-scattered field is c_eff =
            // a + b_b, not a + b. The pre-#305 line added BOTH scattering
            // species to the extinction, which charges the glow for every
            // forward-scattered photon -- and Petzold particulates are
            // 98.3% forward, so a coastal column went four times too dark.
            // Same law as transmittance() above and as the water-leaving
            // closed form at the MAT_WATER interface, which is the point.
            float3 sigma_t = ptWaterEffAtten();
            float  sigma_t_lo = min(sigma_t.r, min(sigma_t.g, sigma_t.b));
            if (any(sigma_s > float3(0.0)) && sigma_t_lo > 0.0) {
                // ln(255): one 8-bit level. Mirrors
                // pt::water::OpticalBlackDepth exactly.
                float t_black = 5.5412635451584 / sigma_t_lo;
                float t_far_w = min(primary_t, t_black);
                int   N_w = clamp(int(water_scatter2.y), 1, 256);
                if (t_far_w > 0.0) {
                    float  dt_w   = t_far_w / float(N_w);
                    // Same stratified form as the air march: (vi + jitter)
                    // covers [0, t_far) exactly, where mid-cell + full-cell
                    // jitter would never sample the first half step -- the
                    // water right in front of the lens, which is the
                    // brightest part of the column.
                    float  jit_w  = randf(seed);
                    float3 v_color_w   = float3(0.0);
                    float  water_n = clamp(water_params0.w, 1.0, 2.4);
                    float  eta_in  = 1.0 / water_n;
                    float3 sun_dir_w = sun_and_mode.xyz;
                    float  g_par   = clamp(water_scatter2.x, -0.99, 0.99);
                    float  depol   = clamp(water_scatter2.w, 0.0, 1.0);
                    float  cone_spread_w =
                        (2.0 * pos_fovtan.w) / float(max(dim.y, 1u));
                    for (int wi = 0; wi < N_w; ++wi) {
                        float t = (float(wi) + jit_w) * dt_w;
                        if (t >= t_far_w) break;
                        float3 sp = primary_ro + primary_rd * t;
                        // Eye transmittance AT THE SAMPLE, in closed form,
                        // rather than as the running product the air march
                        // uses. Not a style difference: the running product
                        // evaluates transmittance at the CELL START while the
                        // sample sits at (wi + jitter) dt, which is an O(dt)
                        // bias no amount of accumulation converges away. The
                        // air march has to accumulate because its sigma
                        // varies with altitude along the ray; water's is
                        // constant, so the exact answer is available for the
                        // same one exp() per step, and the quadrature becomes
                        // second-order in dt instead of first. Pinned by the
                        // convergence ladder in tests/pt_water_optics_test.cpp.
                        float3 trans_eye_w = exp(-sigma_t * t);
                        PtRayCone cone_w =
                            ptConeMake(cone_spread_w * t, cone_spread_w);
                        float sun_elev = elevationAt(sp, sun_dir_w);
                        if (sun_elev > 0.0) {
                            // Snell, into the water. Light propagates along
                            // -sun_dir; refract it through the local
                            // horizontal and negate to get the direction a
                            // submerged point must LOOK to see the sun.
                            // Entering the denser medium cannot TIR, so the
                            // only zero-length result would come from a sun
                            // exactly on the horizon, which sun_elev > 0
                            // has already excluded.
                            float3 up_w = localUp(sp);
                            float3 into = refract(-sun_dir_w, up_w, eta_in);
                            if (dot(into, into) > 1e-6) {
                                float3 sun_w   = normalize(-into);
                                float3 sun_jit = sampleSunDisc(sun_w, seed);
                                float3 trans_w = transmittance(
                                    sp, sun_jit, 1e30, shutter_t01, cone_w,
                                    primary_medium);
                                if (any(trans_w > float3(0.0))) {
                                    // Incident light propagates along
                                    // -sun_jit and leaves toward
                                    // -primary_rd, so the phase cosine is
                                    // dot(primary_rd, sun_jit) -- the same
                                    // orientation the air march derives
                                    // above, and forward scatter therefore
                                    // peaks looking toward the sun.
                                    float mu_w = dot(primary_rd, sun_jit);
                                    float3 in_scatter_w =
                                          sigma_s_mol
                                            * ptPhaseRayleighDepol(mu_w, depol)
                                        + float3(sigma_s_par)
                                            * ptPhaseHenyeyGreenstein(mu_w,
                                                                      g_par);
                                    // #280 renamed this: the value is an
                                    // IRRADIANCE, and the multiply below is
                                    // by a scattering coefficient and a
                                    // phase with no solid angle, which is
                                    // the same dimensional argument the air
                                    // march and the surface NEE make. See
                                    // src/renderer/Atmosphere.h section 1.
                                    // #305 CORRECTS THE REFRACTION
                                    // JACOBIAN. The comment above claimed
                                    // "the n^2 cancels exactly" from
                                    // L_w = L_a (1-F) n^2 against
                                    // Omega_w = Omega_a / n^2. The second
                                    // half is wrong: differentiating
                                    // Snell gives
                                    // Omega_w / Omega_a = mu_a / (n^2 mu_s),
                                    // not 1 / n^2, so the beam irradiance
                                    // in water is
                                    //   E_perp,w = E_perp,a (1-F) mu_a/mu_s
                                    // and the missing factor is mu_a/mu_s.
                                    // It is a pure dimming, 1 at the zenith
                                    // and 0.66 at a grazing sun, and it is
                                    // exactly what makes the horizontal
                                    // downwelling irradiance obey the
                                    // textbook E_d(0-) = E_d(0+) (1 - F).
                                    // Same factor, same spelling, in the
                                    // #305 water-leaving source at the
                                    // MAT_WATER interface -- which is what
                                    // lets a camera just above and just
                                    // below the surface agree.
                                    float  mu_s_w = dot(up_w, sun_w);
                                    float3 sun_rad_w =
                                        sunIrradianceAt(sp, sun_dir_w, sun_elev)
                                        * (sun_elev / max(mu_s_w, 1e-4));
                                    v_color_w += sun_rad_w * trans_w
                                               * in_scatter_w * dt_w
                                               * trans_eye_w;
                                }
                            }
                        }
                    }
                    radiance += v_color_w;
                }
            }
        }
        // --- end Planetary P7 underwater in-scatter -------------------------
#endif  // PT_WATER_ENABLED

        // --- Wave 8 spectral (#27) ---
        // Fold this path's hero-wavelength radiance into the frame estimate.
        // With spectral OFF, hero_lambda < 0 so we take the verbatim
        // pre-#27 accumulation (no multiply) -- bit-stable goldens. With
        // spectral ON, weight the path's radiance by the wavelength's
        // normalised CIE sRGB response so the spp-average (and the temporal
        // accumulator at 1 spp) reconstructs the dispersed colour image.
        if (hero_lambda >= 0.0) {
            frame_radiance += radiance * spectral_weight;
            // B10 SIGMA demod: weight the captured sun terms by the same
            // hero-wavelength CIE response so the spp-average matches the
            // sun contribution folded into frame_radiance.
            frame_sun_shadowed += sample_sun_shadowed * spectral_weight;
            frame_sun_direct   += sample_sun_direct   * spectral_weight;
        } else {
            frame_radiance += radiance;
            frame_sun_shadowed += sample_sun_shadowed;
            frame_sun_direct   += sample_sun_direct;
        }
        // --- end ---
    }
    frame_radiance /= float(sample_count);
    frame_sun_shadowed /= float(sample_count);
    frame_sun_direct   /= float(sample_count);

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
        // Kill NaN/Inf always (they poison the running mean forever). But do
        // NOT clamp negative channels in spectral mode: the hero-wavelength
        // sRGB weights are legitimately negative for out-of-gamut
        // monochromatic light, and those signed lobes must survive into the
        // temporal mean so they cancel there. Zeroing them per frame biases
        // the estimator warm/bright (~+40% red). Negatives are clamped only
        // at display (hdr_src below), leaving the accumulator unbiased.
        bool spectral_on = (spectral_params.x > 0.5);
        float3 clean = spectral_on ? frame_radiance : max(frame_radiance, float3(0.0));
        frame_radiance = select(bad, float3(0.0), clean);
    }
    // (No frame_stars sanitisation -- accum_stars was retired with the
    // EMA-based star_split architecture. Stars+sun+moon are now
    // re-evaluated stateless per frame in StarsComposite.slang post-
    // denoise; no shader-side history to keep finite.)

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
        // Note (issue #46 celestials composite): when
        // composite_celestials_active was set above, the primary-miss
        // code subtracted starsOnly + sunDisc + moonDisc from
        // `radiance`, so frame_radiance carries sky-without-celestials.
        // The engine's StarsComposite kernel re-adds them post-denoise
        // (see shaders/StarsComposite.slang). With the gate off,
        // frame_radiance carries the full sky exactly as the legacy
        // path did.
        // Clamp negatives here, post-reconstruction: spectral mode lets
        // signed CMF lobes into the accumulator (see the sanitiser above),
        // so guard the tonemapper / denoiser / HDR aux from ever seeing
        // negative colour. No-op for non-spectral (already >= 0).
        float3 hdr_src = max((denoiser_enabled != 0u) ? frame_radiance : avg,
                             float3(0.0));
        // B10 SIGMA sun-shadow demodulation (issue #115). Remove the
        // primary-hit direct-sun contribution from the DENOISER INPUT so
        // SVGF never sees -- and so never spatially bleeds -- the hard
        // sun-shadow edge on flat ground. SigmaShadow.slang re-adds the
        // unshadowed sun S post-denoise, modulated by the separately
        // bilateral-denoised visibility, giving a crisp shadow line while
        // leaving point / spot / area lights and indirect GI in the umbra
        // untouched (the old `hdr *= vis` composite multiplied ALL of them
        // by the sun visibility, blacking out shadow interiors -- the bug
        // this fixes). Gated on write_shadow_vis (Metal SIGMA path only;
        // requires the HDR pipeline + procedural sun) and the HDR-pipeline
        // branch below, so the LDR pipeline and every non-Metal / demod-off
        // path keep frame_radiance verbatim. accum_hdr (auto-exposure
        // metering + denoiser-off display) is NOT demodulated -- it always
        // carries the correctly-shadowed sun.
        if (denoiser_enabled != 0u && write_shadow_vis != 0u &&
            w2j_row0.w > 0.5) {
            hdr_src = max(hdr_src - frame_sun_shadowed, float3(0.0));
        }
        if (w2j_row0.w > 0.5) {
            denoise_color[tid] = float4(hdr_src, 1.0);
        } else {
            // --- Wave 9 tonemap --- operator from tonemap_params.x (op 0
            // == acesTonemap, byte-identical to the previous default).
            float3 mapped = tonemapDispatch(hdr_src * exposure, tonemap_params.x);
            denoise_color[tid] = float4(mapped, 1.0);
        }
    }
    // Cloud-transmittance G-buffer (issue #46 follow-up). DIFFERENT
    // gate from denoise_color above: write_hdr_aux fires for both
    // denoiser-active AND bloom-without-denoiser, but cloud_trans_tex
    // is only allocated for the denoiser-active path. Gate on
    // denoiser_enabled to avoid writing to an unbound storage texture
    // when r_bloom=1 / r_denoiser=off. frame_cloud_trans defaults to
    // 1.0 (no cloud, no occlusion) and is overwritten by the volumetric
    // cloud march on sample 0 when the layer is active.
    //
    // Wave 7 (#24): when r_clouds_mode == procedural_raymarched the
    // CloudsRaymarch compute pre-pass already wrote cloud_trans_tex
    // with the camera-ray transmittance from its raymarch. PathTrace
    // must NOT overwrite that value here -- the pre-pass's number
    // is the authoritative cloud transmittance in that mode.
    // StarsComposite reads cloud_trans_tex identically in both modes.
    if (denoiser_enabled != 0u && clouds_runtime.x == 0u) {
        cloud_trans_tex[tid] = frame_cloud_trans;
    }
    // Sun-NEE shadow-visibility G-buffer (issue #115). Same gating
    // shape as cloud_trans_tex above: write_shadow_vis is set by the
    // engine when r_shadow_demod is on AND a denoiser is active AND
    // the shadow_vis_buf is actually allocated. Off in the legacy-
    // shadow path (r_shadow_demod 0) so PathTrace doesn't pay for an
    // extra storage-buffer write per pixel when the SigmaShadow chain
    // is disabled. frame_shadow_vis is the sum over spp samples (each
    // sample contributes its primary-hit shadow-ray transmittance,
    // with sky pixels + sun-below-horizon samples contributing 1.0).
    // Average back to [0, 1] before writing so the downstream
    // SigmaShadow kernel reads a true visibility, not a count.
    //
    // B10 packed layout (issue #115): the buffer holds FOUR floats per
    // pixel now, not one -- [4i+0] = visibility, [4i+1..3] = the
    // unshadowed direct-sun radiance S (RGB). SigmaShadow.slang bilateral-
    // filters the visibility and composites `hdr += S * vis_denoised`, so
    // only the direct-sun term is re-modulated by the sharp visibility;
    // everything else (point/spot/area lights, indirect GI) is left
    // exactly as the radiance denoiser resolved it. The engine sizes
    // shadow_vis_buf at width*height*4 floats to match.
    if (write_shadow_vis != 0u) {
        float vis_mean = frame_shadow_vis / float(sample_count);
        // Sanitize like the radiance path does. The transmittance call
        // is clamped saturate() per-sample above, but the average
        // could touch NaN if a sample slipped through (e.g. a future
        // transmittance variant returning Inf on a degenerate ray) --
        // snap to 1.0 so the worst case is "this pixel doesn't get
        // shadow-demodulated", not "this pixel goes black after
        // composite multiplies by NaN".
        if (isnan(vis_mean) || isinf(vis_mean)) vis_mean = 1.0;
        vis_mean = saturate(vis_mean);
        // Sanitize S the same way: a NaN/Inf here would be added straight
        // into the post-denoise HDR. Clamp negatives to 0 (S is radiance).
        float3 sun_s = frame_sun_direct;
        bool3 sbad = isnan(sun_s) || isinf(sun_s);
        sun_s = select(sbad, float3(0.0), max(sun_s, float3(0.0)));
        uint idx = tid.y * dim.x + tid.x;
        shadow_vis_buf[4u * idx + 0u] = vis_mean;
        shadow_vis_buf[4u * idx + 1u] = sun_s.r;
        shadow_vis_buf[4u * idx + 2u] = sun_s.g;
        shadow_vis_buf[4u * idx + 3u] = sun_s.b;
    }

    // Swapchain write. Apply the OETF on the SPIR-V path so the OS's
    // EOTF-on-display gives back the original linear values. On Metal
    // the swapchain's BGRA8Unorm_sRGB pixel format encodes implicitly
    // on Apple silicon storage writes -- nothing for the shader to do.
    // --- Wave 9 tonemap --- operator from tonemap_params.x (op 0 ==
    // acesTonemap so the inline denoiser-off swapchain write is unchanged).
    float3 ldr = tonemapDispatch(avg * exposure, tonemap_params.x);
#if defined(PT_TARGET_SPIRV)
    ldr = srgb_oetf(ldr);
#endif
    output[tid] = float4(ldr, 1.0);
}

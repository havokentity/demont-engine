// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Procedural raymarched clouds (Wave 7 issue #24).
//
// Production-style cloudscape that runs as a dedicated pre-pass BEFORE
// PathTrace.slang's main dispatch. Provides a fast alternative to the
// inline path-traced cloud march that lives in PathTrace.slang's main()
// (selected via the `r_clouds_mode` cvar):
//
//   r_clouds_mode = pathtraced           -> existing inline march runs
//                                           (bit-for-bit preserved).
//   r_clouds_mode = procedural_raymarched -> this kernel handles all
//                                           cloud sampling; PathTrace's
//                                           inline march is skipped via
//                                           a runtime branch gate.
//
// Why a dedicated pass:
//   Real-time renderers do not pure-path-trace clouds. Schneider's
//   Horizon / Death Stranding cloud tech, Decima, Frostbite and most
//   film pipelines raymarch the cloud layer in a separate pass with
//   light-marching for transmittance, then composite into the path-
//   traced output. Path-tracing low-density volumetric media is
//   catastrophically expensive: rays bounce dozens of times through
//   99% transparent space. Raymarch + light-march at controlled step
//   counts is 10-50x faster and visually indistinguishable when
//   tuned properly.
//
// Algorithm (per primary ray, one dispatch invocation = one screen pixel):
//   1. Reconstruct the primary ray from the camera basis (matches
//      PathTrace.slang's pixel-center -> ray reconstruction so the
//      composite lines up bit-exact with the rest of the frame).
//   2. Intersect the slab [base_y, top_y] -> [t_in, t_out].
//   3. March uniformly through the slab, evaluating cloud_density_at
//      at each sample (imported from PathTraceCloud.slang module).
//   4. At each non-empty sample, light-march 6 steps toward the sun
//      (Beer-Lambert: T_sun = exp(-sum(sigma_t * dt_sun))). Multiply
//      by HG phase + powder term + altitude-boosted sun radiance to
//      get the in-scatter contribution; multiply by camera-ray
//      transmittance accumulated so far. Accumulate into the running
//      colour.
//   5. Advance camera-ray transmittance: T_cam *= exp(-sigma_t * dt).
//   6. Write { rgb = in_scatter, a = 1 - T_cam } to clouds_color_tex
//      and T_cam to cloud_trans_tex (StarsComposite reads this slot
//      identically in both modes).
//
// Light-march: 6 fixed steps with exponential growth (1.5x per step).
// Schneider/Decima tuning -- finely sampled close to the cloud sample
// for accurate self-shadowing, coarsely sampled far away because the
// outer cloud body's optical depth integrates over kilometres where
// individual steps don't matter for the final transmittance.
//
// The composite math in CloudsRaymarchComposite.slang (or folded into
// StarsComposite -- we keep it separate for clarity):
//   final_hdr = clouds_color.rgb + (1.0 - clouds_color.a) * pt_radiance
// Pre-multiplied alpha so dense clouds (alpha -> 1) replace the path-
// traced scene radiance, thin clouds (alpha -> 0) leave it untouched.
//
// Temporal accumulation: optional polish, controlled by tem_alpha.
// When > 0 the kernel EMAs against the previous frame's
// clouds_color_tex (which the engine ping-pongs); when 0 the kernel
// just overwrites.
//
// Half-res rendering: optional polish, controlled by resolution_scale.
// Caller dispatches at (W*scale, H*scale) groups so each kernel
// invocation owns a (1/scale)x(1/scale) tile and writes the
// upper-left pixel. The engine's composite step bilateral-upsamples
// to full-res using depth as the edge guide.
//
// MSL SLOT NOTICE: this kernel has TWO texture bindings (clouds_color
// at slot 0, cloud_trans at slot 1) and ONE buffer (Push at MSL buf 0).
// No AS / SSBO declarations.
//
// vk::image_format: clouds_color_tex is RGBA16F (pre-multiplied
// radiance + alpha), cloud_trans_tex is R32F (single-channel
// transmittance scalar) -- same format StarsComposite expects.

import PathTraceCloud;

[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> clouds_color_tex;
[[vk::binding(1, 0), vk::image_format("r32f")]]    RWTexture2D<float>  cloud_trans_tex;
// Previous-frame clouds_color (for temporal EMA). Engine ping-pongs
// two RGBA16F textures and binds the OTHER one here each frame; when
// temporal_alpha == 0 we ignore it (avoids the read entirely).
[[vk::binding(2, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> clouds_color_prev;

#ifdef PT_TARGET_SPIRV
// Vulkan path. NVIDIA's maxPushConstantsSize is 256B and the engine's
// shared pipeline layout reserves the first 112B for hw push constants
// (VulkanDevice::kPushSplitOffset), with the remainder spilling into
// a per-frame Frame UBO. The host CloudsRaymarchPush struct is
// contiguous 240B; VulkanCommandBuffer::Dispatch byte-splits it:
// bytes [0..112) -> vkCmdPushConstants, bytes [112..240) -> Frame UBO.
[[vk::push_constant]]
cbuffer Push {
    // Camera basis (matches PathTrace.slang's leading fields so the
    // engine can memcpy from PtPush).
    float4 pos_fovtan;     // .xyz origin, .w fovYTan
    float4 fwd_aspect;     // .xyz forward, .w aspect
    float4 right_xyz;
    float4 up_xyz;
    float4 sun_and_mode;   // .xyz sun unit, .w sky mode
    float4 moon_dir_phase; // .xyz moon unit, .w phase rad
    float4 sun_extra;      // .x size .y dist-ratio .z vflat .w planet R
};

[[vk::binding(14, 0)]]
cbuffer Frame {
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 clouds_p4;
    // .x vol_density_scale, .y vol_phase_g_cloud, .z vol_multiscatter_oct,
    // .w sun_radiance_scale (we collapse sun_rad to a scalar through the
    // exposure_pad chain on the host side -- simpler than re-deriving it
    // from sky mode every kernel).
    float4 raymarch_params;
    // .x temporal_alpha (0 disables EMA, [0,1]), .y resolution_scale
    // (1.0 = full-res; 0.5 = half-res; engine clamps).
    // .z light_march_steps (clamped 1..16 in shader), .w sample_count
    // (clamped 8..128 in shader).
    float4 raymarch_quality;
    // Sun & moon radiance (linear RGB). These are pre-computed by the
    // engine to match PathTrace.slang's sun_rad/moon_rad derivation so
    // the pre-pass matches the inline march's lighting.
    float4 sun_radiance;
    float4 moon_radiance;
    // Sky-tint ambient (PathTrace's sky_ambient term, baked into a
    // single colour for the procedural-sky case).
    float4 sky_ambient;
    uint   frame_index;
    uint   composite_active;  // 0 -> kernel writes pass-through (alpha 0)
    uint   _pad0;
    uint   _pad1;
};

#else // !PT_TARGET_SPIRV -- Metal MSL path

// Metal lowers vk::push_constant to setBytes with no hw size cap, so we
// keep the full struct contiguous; field order must match the SPIR-V
// Push + Frame declarations above byte-for-byte (the host pushes one
// contiguous 240B blob).
[[vk::push_constant]]
cbuffer Push {
    float4 pos_fovtan;
    float4 fwd_aspect;
    float4 right_xyz;
    float4 up_xyz;
    float4 sun_and_mode;
    float4 moon_dir_phase;
    float4 sun_extra;
    float4 clouds_p1;
    float4 clouds_p2;
    float4 clouds_p3;
    float4 clouds_p4;
    float4 raymarch_params;
    float4 raymarch_quality;
    float4 sun_radiance;
    float4 moon_radiance;
    float4 sky_ambient;
    uint   frame_index;
    uint   composite_active;
    uint   _pad0;
    uint   _pad1;
};

#endif // PT_TARGET_SPIRV

// Per-pixel deterministic hash for jitter; mirrors PathTrace.slang's
// pcgHash so two invocations on the same pixel produce identical noise.
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randf(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

// Henyey-Greenstein single-lobe phase function. Returns the SAME math
// as PathTrace.slang's inline cloud march so the visual character of
// the two modes matches: forward-scatter (g ~ +0.7..+0.9) gives the
// silver-lining lobe around the sun.
float hgPhase(float c_t, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * c_t;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(denom, 1e-4), 1.5));
}

// Light-march toward `light_dir` from `pos`. Returns transmittance T
// (Beer-Lambert) accumulated over light_steps with exponential growth
// (multiplier 1.5 per step -- Schneider production tuning). t_max
// caps the march distance: we don't need to integrate past the cloud
// top because the field is zero above it. Returns 1.0 when the layer
// is entirely above the march or empty.
float lightMarch(float3 pos, float3 light_dir,
                 float4 cp1, float4 cp2, float4 cp3, float4 cp4,
                 float vol_dscale,
                 int   light_steps) {
    // Cap the light-march at the cloud layer top + a small slack so
    // a sample near the top still gets a few steps before exiting.
    float top_y = cp1.y;
    float t_max = 0.0;
    // Step the ray to the cloud-layer top (or to the bottom if the
    // sun is below horizon -- in that case we just early-out, no
    // useful light gets through).
    if (light_dir.y > 1e-4) {
        t_max = max(0.0, (top_y - pos.y) / light_dir.y);
    } else {
        // Sun grazing the horizon or below: light-march pointlessly
        // long, so cap at 2km and let the optical depth shut it down.
        t_max = 2000.0;
    }
    if (t_max <= 0.0) return 0.0;
    // Six steps with exponential growth: dt[i] = dt0 * 1.5^i.
    // The geometric sum of 1.5^0 .. 1.5^(N-1) = (1.5^N - 1) / 0.5.
    // We solve for dt0 such that sum(dt0 * 1.5^i) = t_max.
    int   N = clamp(light_steps, 1, 16);
    float gr = 1.5;
    float geo_sum = (pow(gr, float(N)) - 1.0) / (gr - 1.0);
    float dt0 = t_max / geo_sum;
    float od = 0.0;
    float dt = dt0;
    float t = 0.0;
    [unroll(16)] for (int i = 0; i < N; ++i) {
        float t_mid = t + 0.5 * dt;
        float3 sp   = pos + light_dir * t_mid;
        // Use fewer fbm octaves on the shadow path (3 vs 6) -- the
        // dominant cost in raymarched clouds is light-march samples,
        // and Beer-Lambert exp(-OD) is forgiving to fbm aliasing.
        // Matches the same trade-off the inline cloud_optical_depth
        // path in PathTrace.slang makes (3 octaves on the shadow ray).
        float dens = cloud_density_at(sp, cp1, cp2, cp3, cp4, 3) * vol_dscale;
        od += max(dens, 0.0) * dt;
        t  += dt;
        dt *= gr;
    }
    return exp(-od);
}

// Powder approximation (Schneider 2015). Models the dark-bottom look
// of real clouds: shaded interior with bright back-lit silver-lining
// rim. The single-scatter Beer-Lambert alone gives a too-uniform soft
// look; powder darkens cells where the light has to travel through
// dense material to reach the camera.
//
//   powder(d) = 1 - exp(-2 * d * dt_layer)
//
// Here d is the local sample density, dt_layer is the camera-march
// step size (cumulative density along the immediate camera-march
// segment). Returns [0, 1]; the in-scatter contribution is multiplied
// by (1 - 0.4 * (1 - powder)) to inject the dark-bottom term without
// killing dim lighting on thin cloud edges.
float powderTerm(float dens, float dt_layer) {
    return 1.0 - exp(-max(dens, 0.0) * dt_layer * 2.0);
}

// Slab intersection: cloud layer is [base_y, top_y].
// Returns (t_in, t_out); when ray misses the slab, t_out < t_in.
float2 slabIntersect(float3 ro, float3 rd, float base_y, float top_y) {
    float ry = rd.y;
    if (abs(ry) > 1e-6) {
        float t_b = (base_y - ro.y) / ry;
        float t_t = (top_y  - ro.y) / ry;
        float t_in  = max(0.0, min(t_b, t_t));
        float t_out = max(t_b, t_t);
        return float2(t_in, t_out);
    } else if (ro.y >= base_y && ro.y <= top_y) {
        return float2(0.0, 100000.0);
    }
    return float2(1.0, 0.0);  // miss
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    clouds_color_tex.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;

    // Composite-active early-out: clouds disabled or mode != raymarched.
    // Engine elides the dispatch entirely in those cases but the guard
    // here costs nothing and protects against init-order races.
    if (composite_active == 0u || clouds_p1.w <= 0.0) {
        clouds_color_tex[tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        cloud_trans_tex[tid.xy]  = 1.0;
        return;
    }

    // Reconstruct primary ray. Matches PathTrace.slang's pixel-center
    // computation (no jitter -- the cloud pre-pass takes one sample per
    // pixel and uses temporal EMA + sub-pixel jitter from the per-frame
    // seed below for sub-pixel coverage). This pre-pass runs ONCE per
    // frame so it cannot afford spp>1 directly; temporal accumulation
    // is the substitute.
    float fovY = pos_fovtan.w;
    float aspect = fwd_aspect.w;
    // Sub-pixel jitter for temporal accumulation. Halton-style 2D
    // Hammersley pattern would be tighter, but for clouds the noise
    // tolerance is very high (the field is low-frequency) and a hash-
    // based jitter is enough to break per-pixel aliasing across frames.
    uint pix_seed = pcgHash(tid.x * 1973u + tid.y * 9277u + frame_index * 26699u);
    float jx = randf(pix_seed) - 0.5;
    float jy = randf(pix_seed) - 0.5;
    // When temporal EMA is off, suppress jitter so the output is
    // deterministic frame-to-frame for the golden test path.
    if (raymarch_params.x <= 0.0) { jx = 0.0; jy = 0.0; }
    float2 uv = (float2(float(tid.x) + 0.5 + jx,
                        float(tid.y) + 0.5 + jy) / float2(dim))
              * 2.0 - 1.0;
    uv.y = -uv.y;   // Slang/Vulkan UV convention: y flipped.
    float3 rd = normalize(fwd_aspect.xyz
                        + right_xyz.xyz * uv.x * fovY * aspect
                        + up_xyz.xyz    * uv.y * fovY);
    float3 ro = pos_fovtan.xyz;

    // Slab intersect against the cloud layer.
    float base_y = clouds_p1.x;
    float top_y  = clouds_p1.y;
    float2 t_io  = slabIntersect(ro, rd, base_y, top_y);
    float t_in   = t_io.x;
    float t_out  = t_io.y;
    if (t_out <= t_in) {
        // Ray misses the cloud slab entirely.
        clouds_color_tex[tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        cloud_trans_tex[tid.xy]  = 1.0;
        return;
    }
    // Cap march span at 30km, matching PathTrace.slang's inline march.
    if (t_out - t_in > 30000.0) t_in = t_out - 30000.0;

    // March parameters.
    float vol_dscale = max(raymarch_params.x, 1e-3);
    float g_cloud    = clamp(raymarch_params.y, -0.95, 0.95);
    int   light_steps = clamp(int(raymarch_quality.z + 0.5), 1, 16);
    int   N = clamp(int(raymarch_quality.w + 0.5), 8, 128);
    float dt = (t_out - t_in) / float(N);
    float t  = t_in;
    // Jitter the first sample by [0, dt) so adjacent pixels' samples
    // are decorrelated -- prevents grid aliasing on the cloud field.
    float jit = randf(pix_seed) * dt;

    float3 sun_dir = sun_and_mode.xyz;
    float3 sun_rad = sun_radiance.xyz;
    float3 moon_dir = moon_dir_phase.xyz;
    float3 moon_rad = moon_radiance.xyz;
    float3 ambient  = sky_ambient.xyz;

    float trans_cam = 1.0;
    float3 in_scatter = float3(0.0);

    // Main march.
    [loop] for (int i = 0; i < 128; ++i) {
        if (i >= N) break;
        float t_mid = t + 0.5 * dt + jit;
        if (t_mid > t_out) break;
        float3 sp = ro + rd * t_mid;
        float dens = cloud_density_at(sp, clouds_p1, clouds_p2, clouds_p3, clouds_p4, 6)
                   * vol_dscale;
        if (dens > 0.0) {
            // Light-march toward sun (and moon if active).
            float c_sun = dot(rd, -sun_dir);
            float ph_sun = hgPhase(c_sun, g_cloud);
            float T_sun = (sun_dir.y > 0.0)
                ? lightMarch(sp, sun_dir, clouds_p1, clouds_p2, clouds_p3, clouds_p4, vol_dscale, light_steps)
                : 0.0;
            float c_moon = dot(rd, -moon_dir);
            float ph_moon = hgPhase(c_moon, g_cloud);
            float T_moon = (moon_dir_phase.y > 0.0)
                ? lightMarch(sp, moon_dir, clouds_p1, clouds_p2, clouds_p3, clouds_p4, vol_dscale, light_steps)
                : 0.0;
            // Powder term: dark-bottom approximation per Schneider 2015.
            float powder = powderTerm(dens, dt);
            // Mix the powder term in at 40% strength so cloud cores
            // don't go pitch black under self-shadow while the dark-
            // bottom rim still reads. Same coefficient Decima uses.
            float powder_mix = lerp(1.0, powder, 0.4);
            // Scattering coefficient -- in-scatter per unit length
            // from each light source weighted by phase. step_w gives
            // the discrete-march weight: dens * dt * T_cam, mirroring
            // the inline cloud march's accumulation rule.
            float step_w = dens * dt * trans_cam;
            float3 contrib = float3(0.0);
            contrib += sun_rad  * ph_sun  * T_sun  * powder_mix;
            contrib += moon_rad * ph_moon * T_moon * powder_mix;
            contrib += ambient;     // omnidirectional fill
            in_scatter += contrib * step_w;
            // Advance camera-ray transmittance.
            trans_cam *= exp(-dens * dt);
            // Early exit: when the camera-ray is essentially opaque
            // any further samples can't change the visible alpha or
            // colour. Big speedup on the inside of thick cumulus.
            if (trans_cam < 0.003) break;
        }
        t += dt;
    }

    // Premultiplied alpha. `alpha = 1 - trans_cam` so the composite
    //   final = clouds.rgb + (1 - clouds.a) * scene_radiance
    // collapses to scene_radiance when there are no clouds (alpha 0)
    // and to clouds.rgb alone when the layer is fully opaque (alpha 1).
    float alpha = saturate(1.0 - trans_cam);
    float4 out_c = float4(in_scatter, alpha);

    // Temporal EMA against prev-frame buffer (engine ping-pongs).
    // raymarch_params.x is the cloud march density scale; for temporal
    // we use raymarch_quality.x = tem_alpha. When 0 we skip the read
    // entirely so a single-frame test (golden) is bit-deterministic.
    float tem_alpha = saturate(raymarch_quality.x);
    if (tem_alpha > 0.0 && frame_index > 0u) {
        float4 prev = clouds_color_prev[tid.xy];
        out_c = lerp(out_c, prev, tem_alpha);
    }

    clouds_color_tex[tid.xy] = out_c;
    cloud_trans_tex[tid.xy]  = trans_cam;
}

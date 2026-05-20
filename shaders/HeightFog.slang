// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Volumetric height fog (wave-9). Physically-based exponential-height-
// falloff atmospheric fog, applied as a dedicated post-radiance composite
// pass -- mirrors the AuroraComposite / CloudsComposite architectural
// shape (stateless, one dispatch per frame on Metal between Denoise() and
// the celestial composites, reads depth_tex + writes the HDR in place).
//
// WHY A DEDICATED PASS (not folded into the megakernel):
//   Keeping the fog out of PathTrace.slang's megakernel keeps the
//   path-tracer build time + golden surface untouched (r_fog defaults
//   off; the engine elides this dispatch entirely so existing goldens
//   stay bit-for-bit identical), and lets the analytic integral run once
//   per pixel post-denoise instead of per-bounce. Fog is a low-frequency,
//   view-dependent atmospheric term -- it does not need to participate in
//   the multi-bounce light transport, so a screen-space composite against
//   the primary-hit depth is both correct and cheap.
//
// ORDERING:
//   Dispatched AFTER the SIGMA shadow demod and BEFORE the celestial /
//   cloud / aurora composites. Fog attenuates the scene radiance that the
//   path tracer resolved (geometry); the sun / moon / stars / clouds are
//   composited on TOP afterward because those bodies sit above the fog
//   layer (the exponential-height density is ~zero at celestial
//   altitudes). For sky pixels (primary miss) the fog still integrates
//   along the ray out to a finite far cap so the horizon hazes over and
//   blends into the sky gradient -- the classic "aerial perspective"
//   look -- while the celestials then layer over that haze.
//
// PHYSICS (per project memory: 1 world unit = 1 metre, real constants):
//   Density profile is the standard exponential atmosphere:
//
//       sigma(y) = sigma0 * exp(-(y - y_base) / H)
//
//   sigma0  [1/m]  extinction coefficient at the fog base height y_base.
//                  Real ground haze: ~1e-3 .. 1e-2 /m (visibility a few
//                  hundred m to a few km). Thick valley fog reaches ~1e-1.
//   y_base  [m]    altitude at which density == sigma0 (sea level / valley
//                  floor). Below it density is clamped to the y_base value
//                  (we don't let the exponential blow up underground).
//   H       [m]    scale height -- the e-folding distance of the density
//                  falloff. Real radiation fog: ~100 m. Haze layers /
//                  marine boundary layer: ~500 .. 1000 m.
//
//   Optical depth along a ray r(s) = ro + s*rd, s in [0, t], with the
//   ray's vertical component d = rd.y:
//
//       tau(t) = integral_0^t sigma(ro.y + s*d) ds
//              = sigma0 * exp(-(ro.y - y_base)/H)
//                       * integral_0^t exp(-s*d/H) ds
//
//   The inner integral is the closed form (Wenzel 2007, "Real-time
//   Atmospheric Effects in Games"; also the GPU Gems analytic-fog
//   derivation). Let k = d / H:
//
//       integral_0^t exp(-k*s) ds = (1 - exp(-k*t)) / k        (k != 0)
//                                 = t                          (k -> 0)
//
//   so
//
//       tau(t) = rho_base * (1 - exp(-k*t)) / k                (k != 0)
//       tau(t) = rho_base * t                                  (k -> 0)
//
//   where rho_base = sigma0 * exp(-(ro.y - y_base)/H) is the density at
//   the eye's height. The k->0 (near-horizontal ray) branch is handled
//   with a Taylor guard so the divide stays well-conditioned.
//
//   Transmittance T = exp(-tau) (Beer-Lambert). The fog in-scatter is
//   the classic single-scatter "fog colour weighted by (1 - T)" with a
//   Henyey-Greenstein phase modulating the brightness toward the sun:
//
//       L_out = T * L_scene + (1 - T) * L_fog
//       L_fog = fog_tint * (ambient + sun_lum * hg(cos_theta, g))
//
//   This is single-scatter (no multiple-scattering term): a multi-scatter
//   approximation buys little for thin atmospheric fog at the brightness
//   range of this engine and is left out for clarity + perf.
//
// BINDINGS (Metal MSL slots assigned in declaration order: hdr_inout ->
// texture(0), depth_tex -> texture(1)). The vk::binding tags follow the
// CloudsComposite / AuroraComposite convention (binding 0 = hdr_inout,
// binding 7 = depth_tex matching PathTrace.slang's G-buffer depth slot).
// This pass needs NO scratch / output texture (it composites in place),
// so it does NOT claim the wave-9-reserved binding 36 / engine slot 17.
// vk::image_format: hdr_inout is RGBA16F, depth_tex is R32F. See
// PathTrace.slang for the NVIDIA Format=Unknown rationale.
[[vk::binding(0, 0), vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_inout;
[[vk::binding(7, 0), vk::image_format("r32f")]]    RWTexture2D<float>  depth_tex;

// MSL SLOT NOTICE: this kernel binds NO buffers and NO acceleration
// structures, so Slang's MSL backend assigns Push (the only buffer-class
// declaration below) to MSL buf(0). MetalCommandBuffer::Dispatch computes
// push_slot = max_bound_buf_slot + 1; with nothing else bound and
// BindComputePipeline clearing prior bindings, max_slot = -1 -> push_slot
// = 0. Match. Same layout rule as AuroraComposite / StarsComposite.

[[vk::push_constant]]
cbuffer Push {
    // Camera basis -- same prefix layout as AuroraComposite / PathTrace's
    // Push, copied host-side from PtPush so the ray reconstruction here
    // matches the path-tracer's primary rays exactly.
    float4 pos_fovtan;     // .xyz = camera origin (m), .w = fovYTan
    float4 fwd_aspect;     // .xyz = forward (unit), .w = aspect ratio (W/H)
    float4 right_xyz;      // .xyz = camera right (unit)
    float4 up_xyz;         // .xyz = camera up (unit)

    // .xyz = unit vector toward the sun. .w = sky mode (unused here;
    // carried for layout symmetry with the other composites).
    float4 sun_and_mode;

    // Fog parameters, packed:
    //   .x = sigma0   extinction at base height [1/m]   (r_fog_density)
    //   .y = y_base   base altitude [m]                 (r_fog_base_y)
    //   .z = H        scale height [m]                  (r_fog_scale_height)
    //   .w = g        Henyey-Greenstein anisotropy      (r_fog_sun_anisotropy)
    float4 fog_params;

    // .xyz = fog colour tint (linear RGB)               (r_fog_color_*)
    // .w   = sun in-scatter luminance scale (engine HDR-linear units).
    //        Fixed host-side; the HG phase modulates this toward the sun.
    float4 fog_tint;

    // .x = 1 -> run the composite; 0 -> early-out (engine also elides
    //      the dispatch when r_fog == 0, so this is belt-and-braces).
    // .y = far-cap distance [m] for sky (primary-miss) pixels: the
    //      analytic integral's upper bound when depth_tex carries the
    //      sky sentinel. Caps the otherwise-infinite horizon integral.
    // .z = ambient sky-tint in-scatter scale (the always-on portion of
    //      L_fog independent of sun direction).
    // .w = reserved.
    float4 fog_runtime;
};

// Henyey-Greenstein single-lobe phase function. Same math as
// CloudsRaymarch.slang's hgPhase so fog forward-scatter brightening
// matches the cloud silver-lining lobe. g in (-1, 1): g>0 forward
// scatter (haze brightens toward the sun), g<0 back scatter.
float hgPhase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(denom, 1e-4), 1.5));
}

// Closed-form optical depth of the exponential-height fog along a ray
// from the eye (height y0) over distance t, with vertical ray component
// d = rd.y. See the file header for the derivation.
//
//   rho_base = sigma0 * exp(-(y0 - y_base) / H)   (density at the eye)
//   k        = d / H
//   tau(t)   = rho_base * (1 - exp(-k*t)) / k     (k != 0)
//            = rho_base * t                       (k -> 0, Taylor)
float fogOpticalDepth(float y0, float d, float t,
                      float sigma0, float y_base, float H) {
    // Density at the eye height. Clamp the height-above-base so a camera
    // far below the fog base doesn't produce an absurd exp() overflow:
    // physically the fog can't be denser than its base value in this
    // single-layer model, so cap rho at the base density.
    float h_above = (y0 - y_base) / max(H, 1e-3);
    float rho_base = sigma0 * exp(-max(h_above, 0.0)); // h<0 -> clamp to rho at base
    // If the camera is below the base, density at the eye is the base
    // value (rho_base above already saturates because max(h_above,0)=0).
    float k = d / max(H, 1e-3);
    // k -> 0: the ray is (near) horizontal, density is ~constant along
    // it, so tau ~ rho_base * t. Use the analytic form away from zero and
    // a 2nd-order Taylor blend near it to avoid catastrophic cancellation
    // in (1 - exp(-k*t)) / k. The threshold 1e-4 keeps the relative error
    // of the analytic branch below the HDR quantisation.
    float kt = k * t;
    float integral;
    if (abs(kt) < 1e-4) {
        // (1 - exp(-x))/k = t * (1 - x/2 + x^2/6 - ...) with x = kt.
        integral = t * (1.0 - 0.5 * kt + (1.0 / 6.0) * kt * kt);
    } else {
        integral = (1.0 - exp(-kt)) / k;
    }
    return rho_base * integral;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 dim;
    hdr_inout.GetDimensions(dim.x, dim.y);
    if (tid.x >= dim.x || tid.y >= dim.y) return;
    if (fog_runtime.x < 0.5) return;          // r_fog off -> no-op

    float sigma0 = fog_params.x;
    if (sigma0 <= 0.0) return;                 // zero density -> no fog
    float y_base = fog_params.y;
    float H      = fog_params.z;
    float g      = clamp(fog_params.w, -0.95, 0.95);

    // Reconstruct the world-space ray direction for this pixel. Same
    // formula as AuroraComposite / the StarsComposite per-sample path;
    // one ray through the pixel centre (fog is large-scale, sub-pixel
    // jitter has no perceivable effect).
    float2 uv = ((float2(tid.xy) + 0.5) / float2(dim)) * 2.0 - 1.0;
    uv.y = -uv.y;
    float3 fwd = fwd_aspect.xyz;
    float3 rd = normalize(fwd
                       + right_xyz.xyz * (uv.x * fwd_aspect.w * pos_fovtan.w)
                       + up_xyz.xyz    * (uv.y * pos_fovtan.w));
    float3 ro = pos_fovtan.xyz;

    // Hit distance along the ray. depth_tex stores CAMERA-SPACE Z
    // (projection of the hit point onto the forward axis), NOT the ray
    // length -- PathTrace.slang writes `depth_view = h0.t * dot(rd0,
    // normalize(fwd))`. To recover the ray length t we divide that Z by
    // the ray's forward cosine: t = depth_z / dot(rd, fwd). For sky
    // pixels PathTrace writes a 1e10 sentinel; we substitute the far-cap
    // distance so the horizon still hazes over instead of integrating to
    // infinity.
    float depth_z = depth_tex[tid.xy];
    float cos_fwd = max(dot(rd, normalize(fwd)), 1e-3);
    float t;
    bool is_sky = depth_z >= 1.0e9;
    if (is_sky) {
        // Far-cap [m] for the sky integral. fog_runtime.y is the cap
        // distance; clamp to a sane floor so a mis-set cvar can't zero
        // the horizon haze.
        t = max(fog_runtime.y, 1.0);
    } else {
        t = depth_z / cos_fwd;
    }
    if (t <= 0.0) return;

    // Optical depth + Beer-Lambert transmittance over [0, t].
    float tau = fogOpticalDepth(ro.y, rd.y, t, sigma0, y_base, H);
    float T = exp(-max(tau, 0.0));
    // Nothing to do if the ray is essentially fog-free (T ~ 1). Saves the
    // composite read+write on the bulk of an above-the-fog frame.
    if (T >= 0.9999) return;

    // Fog in-scatter colour. Ambient sky-tint term (always on) plus a
    // sun forward-scatter lobe (Henyey-Greenstein) that brightens the
    // fog toward the sun -- the "sun-side glows" look the acceptance
    // criteria call for. cos_theta is the angle between the view ray and
    // the direction TOWARD the sun.
    float3 tint = fog_tint.rgb;
    float sun_lum = fog_tint.w;            // sun in-scatter luminance scale
    float ambient = fog_runtime.z;         // ambient sky-tint scale
    float3 sun = sun_and_mode.xyz;
    float cos_theta = dot(rd, sun);
    // Day gate: fog should not pick up a sun lobe when the sun is below
    // the horizon (night). Smooth ramp matches the celestial-composite
    // day fade so the sun glow fades out at twilight rather than popping.
    float sun_above = smoothstep(-0.10, 0.05, sun.y);
    float phase = hgPhase(cos_theta, g);
    // Phase is normalised to integrate to 1 over the sphere (peak ~ a few
    // for g~0.7); multiply by 4*pi so the forward lobe reads as a
    // perceptible brightening rather than a fraction. The sun term is
    // gated by sun_above so it vanishes at night.
    float sun_inscatter = sun_lum * phase * (4.0 * 3.14159265358979) * sun_above;
    float3 L_fog = tint * (ambient + sun_inscatter);

    // Composite: L_out = T * L_scene + (1 - T) * L_fog (Beer-Lambert
    // single-scatter). For sky pixels this hazes the horizon; the
    // celestial / cloud composites run AFTER this pass and layer over
    // the hazed sky.
    float3 L_scene = hdr_inout[tid.xy].rgb;
    float3 out_rgb = T * L_scene + (1.0 - T) * L_fog;

    // Sanitize: never emit NaN/Inf or negatives (defence-in-depth,
    // matching AuroraComposite's policy). The analytic path can't
    // actually produce non-finite values, but the guard is free.
    bool3 bad = isnan(out_rgb) || isinf(out_rgb);
    out_rgb = select(bad, L_scene, max(out_rgb, float3(0.0)));

    hdr_inout[tid.xy] = float4(out_rgb, 1.0);
}

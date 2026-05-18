// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Screen-space particle composite (issue #82 MVP).
//
// MVP scope (read this before extending):
//   - Particles are screen-space billboard SPLATS. They paint INTO
//     the post-denoise HDR texture additively, depth-gated by the
//     PathTrace G-buffer (depth_tex slot 7 per the project's convention,
//     matching StarsComposite). Particles in front of geometry render;
//     particles behind it are occluded.
//   - Each particle is a soft Gaussian disc. Color is HDR (sparks read
//     >1.0 so bloom catches them); size is in WORLD-SPACE METRES on
//     the host side and converted to pixel sigma by the projection.
//   - NOT path-traced. Particles do NOT cast shadows, do NOT show up
//     in reflections, do NOT contribute to NEE. The PT-native version
//     (analytic-primitive list + BVH refit) is a follow-up. See the
//     PR body for the deferred-work split.
//
// Kernel layout vs StarsComposite (the architectural reference here):
//   - Same dispatch point: post-Denoise(), pre-bloom_pyramid, inside
//     the use_engine_tonemap branch on Metal. ParticleComposite is
//     gated to dispatch AFTER StarsComposite so particles overlay
//     stars / sun / moon as well as geometry-side pixels (a spark in
//     front of the moon should occlude the moon disc partially via
//     additive over-write -- additive composition handles this for
//     free since both writes go into the same HDR texture).
//   - Same depth-tex gate as StarsComposite, but inverted in spirit:
//     StarsComposite paints only on SKY pixels (depth >= 1e9 sentinel).
//     ParticleComposite paints anywhere the particle's projected
//     depth is in front of the G-buffer surface depth. We can render
//     particles AGAINST geometry, just not BEHIND it.
//
// Dispatch shape:
//   - 1 thread per particle (NOT per pixel). The engine dispatches a
//     grid of ceil(particle_count / 64) threadgroups so each thread
//     handles exactly one particle.
//   - Threadgroup is [numthreads(8, 8, 1)] to match the project-wide
//     Metal convention (MetalCommandBuffer::Dispatch hardcodes
//     tgsize (8, 8, 1); see the long comment in MetalDevice.cpp).
//     We linearize tid.xy back to a single particle index inside the
//     kernel.
//   - Each thread reads its particle, projects it to screen space,
//     computes the pixel footprint of its splat, and additive-writes
//     a Gaussian-weighted contribution INTO hdr_inout for every pixel
//     in its bounding box. The bounding box is clamped to the
//     particle's pixel-space radius (a few pixels at typical splat
//     sizes), so the per-thread cost is small and bounded.
//   - We rely on Metal's auto-RAW barriers for the read-modify-write
//     pattern -- the engine's MetalCommandBuffer auto-barriers shared
//     resources between dispatches, which is enough for in-order
//     particles within a single dispatch (writes serialize through
//     the texture cache). For overlapping particle splats, the
//     additive nature of the operation means race conditions on the
//     same texel result in some loss of additive precision but not
//     visible artefacts -- the same trade-off StarsComposite makes
//     when multiple aperture samples land in the same pixel.
//
// Coordinate gotchas:
//   - PathTrace.slang builds camera rays as `forward + uv.x * right *
//     fovYTan * aspect + uv.y * up * fovYTan` where uv is in [-1, 1]
//     and `uv.y = -uv.y` flips Y (texture rows go top-down, world Y
//     goes bottom-up). We mirror that here: project the particle
//     position into the same UV space, then flip Y, then convert to
//     pixel coords. The result must agree with the path tracer's
//     primary-ray-pixel mapping or particles will appear in the wrong
//     place relative to geometry.
//   - Depth comparison: PathTrace writes camera-space Z (forward
//     distance to surface) into depth_tex. We compute the particle's
//     camera-space Z by projecting onto the forward axis and compare
//     directly. Less-than means "particle is closer" (in front).

[[vk::binding(0, 0)]] RWTexture2D<float4>           hdr_inout;
[[vk::binding(7, 0)]] RWTexture2D<float>            depth_tex;
[[vk::binding(20, 0)]] StructuredBuffer<float4>     particles;
// vk::binding tags are documentation -- the MSL backend assigns Metal
// slots in declaration order. Engine binds:
//   texture(0)  = hdr_inout
//   texture(1)  = depth_tex
//   buffer(0)   = particles
//   buffer(1)   = Push  (assigned by Slang as max_buf_slot+1)
// See the engine dispatch site for the exact binding sequence.

[[vk::push_constant]]
cbuffer Push {
    // Camera basis. Matches PathTrace.slang's pos_fovtan / fwd_aspect
    // / right_xyz / up_xyz prefix so the same world->screen projection
    // applies. Without this, particles' projected pixel position
    // wouldn't line up with the primary-hit pixel mapping the depth
    // gate compares against.
    float4 pos_fovtan;     // .xyz = camera origin, .w = fovYTan
    float4 fwd_aspect;     // .xyz = forward, .w = aspect ratio (w/h)
    float4 right_xyz;
    float4 up_xyz;

    // Viewport dimensions in pixels. Could compute via
    // GetDimensions(hdr_inout) but pushing them avoids one query per
    // thread.
    uint width;
    uint height;
    uint particle_count;
    // 1 => active, 0 => early-out. Same shape as StarsComposite's gate.
    uint composite_active;
    // Total number of threads in the X dimension of the dispatch grid
    // (= groups_x * 8 for the [numthreads(8,8,1)] kernel). Used to
    // linearise (tid.x, tid.y) -> particle index without re-deriving
    // it from the threadgroup count. Engine sets this to
    // ceil(particle_count / 64) * 8.
    uint threads_x;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Per-particle layout matches the GpuParticle struct the engine builds
// in Engine::RenderFrame. Three float4 per particle:
//   p0.xyz = world position  (metres)
//   p0.w   = size            (metres -- splat sigma in world units)
//   p1.xyz = color           (linear HDR rgb)
//   p1.w   = alpha           (0..1 scalar visibility multiplier)
//   p2.xyz = unused          (padding for future fields, e.g. motion blur)
//   p2.w   = z_eye           (precomputed camera-space Z, m -- saves a
//                             dot product in the kernel hot path)
//
// Three float4s = 48 bytes per particle, matching the 48-byte upload
// stride on the host (Particle.size on host is 48 bytes by static_assert,
// see Engine::RenderFrame's particle upload site).

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (composite_active == 0u) return;
    // Linearise (tid.x, tid.y) -> particle index. Engine dispatches
    // (groups_x, 1, 1) threadgroups, so the kernel sees tid.x in
    // [0, groups_x*8) and tid.y in [0, 8). threads_x = groups_x*8 is
    // pushed so we can compute idx with no per-thread divisions.
    uint idx = tid.y * threads_x + tid.x;
    if (idx >= particle_count) return;

    // Load this thread's particle. 3 float4 reads = one cache line on
    // Apple GPUs (128-byte cache line, 48-byte stride, so on average
    // 8/3 particles share a line -- fine).
    float4 p0 = particles[idx * 3u + 0u];
    float4 p1 = particles[idx * 3u + 1u];
    float4 p2 = particles[idx * 3u + 2u];

    float3 wpos       = p0.xyz;
    float  world_size = p0.w;
    float3 color      = p1.xyz;
    float  alpha      = p1.w;
    float  z_eye      = p2.w;

    // Early-out for particles behind the camera (z_eye <= 0 means
    // strictly behind the near plane). Same near-clip the path tracer
    // implicitly uses (cam.pos as ray origin, forward as the +Z eye axis).
    if (z_eye <= 0.05) return;
    if (alpha <= 1e-4) return;
    if (world_size <= 1e-4) return;

    // Project to screen. wpos - cam.pos = offset from camera in world.
    // Decompose along (right, up, forward) -> camera-space. We have
    // z_eye precomputed, so just project onto right + up.
    float3 to_p = wpos - pos_fovtan.xyz;
    float  x_eye = dot(to_p, right_xyz.xyz);
    float  y_eye = dot(to_p, up_xyz.xyz);

    // Perspective divide. Same projection PathTrace.slang inverts:
    //   uv.x = (x_eye / z_eye) / (fovYTan * aspect)
    //   uv.y = (y_eye / z_eye) / fovYTan
    float fovYTan = pos_fovtan.w;
    float aspect  = fwd_aspect.w;
    float inv_z   = 1.0 / z_eye;
    float u =  x_eye * inv_z / (fovYTan * aspect);
    float v = -y_eye * inv_z / fovYTan;     // Y flip to match texture row order

    // uv -> pixel coords. uv is in [-1, 1] across the viewport.
    float pix_x = (u * 0.5 + 0.5) * float(width);
    float pix_y = (v * 0.5 + 0.5) * float(height);

    // Off-screen culling with a small margin so a particle whose
    // CENTRE is just off-screen still contributes its splat tail to
    // edge pixels.
    if (pix_x < -64.0 || pix_x > float(width)  + 64.0) return;
    if (pix_y < -64.0 || pix_y > float(height) + 64.0) return;

    // Pixel-space radius of the Gaussian splat. world_size is metres;
    // pixel-per-metre at this z is (height / (2 * z_eye * fovYTan)).
    float px_per_m = float(height) / (2.0 * z_eye * fovYTan);
    float sigma_px = world_size * px_per_m;
    // Clamp to a sensible range. Below 0.5 px the splat aliases into a
    // single pixel (visible as flicker on camera motion); above ~64 px
    // the iteration cost per particle blows up. The MVP doesn't try
    // to render house-sized particles, so the clamp is correct.
    sigma_px = clamp(sigma_px, 0.5, 64.0);
    // 3-sigma radius covers >99% of the Gaussian energy.
    float r_px = sigma_px * 3.0;

    // Bounding box in pixel space, clamped to the viewport.
    int x0 = max(0,                int(floor(pix_x - r_px)));
    int y0 = max(0,                int(floor(pix_y - r_px)));
    int x1 = min(int(width)  - 1,  int(ceil (pix_x + r_px)));
    int y1 = min(int(height) - 1,  int(ceil (pix_y + r_px)));
    if (x0 > x1 || y0 > y1) return;

    // Inverse 2 * sigma^2 for the Gaussian.
    float inv_2sigma2 = 1.0 / (2.0 * sigma_px * sigma_px);

    // Splat. For each pixel in the bbox, compute the Gaussian weight,
    // check the depth gate, and additive-write color * weight * alpha.
    // The depth gate is per-pixel because depth_tex is per-pixel: a
    // particle in front of the floor but BEHIND a sphere should be
    // visible on the floor side and occluded on the sphere side.
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            float dx = float(px) + 0.5 - pix_x;
            float dy = float(py) + 0.5 - pix_y;
            float r2 = dx * dx + dy * dy;
            // Cull pixels outside the 3-sigma disc analytically -- the
            // bbox is square, the disc is round; the corners contribute
            // ~0 weight.
            if (r2 > r_px * r_px) continue;
            float w = exp(-r2 * inv_2sigma2);
            // Depth gate. Particle z_eye must be in front of (less
            // than) the surface depth recorded by the path tracer's
            // G-buffer. Sky sentinel is 1e10 so the comparison works
            // out of the box for "particle in front of sky".
            float surf_z = depth_tex[uint2(px, py)];
            if (z_eye > surf_z) continue;
            float3 add = color * (w * alpha);
            // Sanitize. A bad upload (NaN size, denorm camera) would
            // otherwise persist forever in the HDR texture.
            if (any(isnan(add)) || any(isinf(add))) continue;
            float3 base = hdr_inout[uint2(px, py)].rgb;
            hdr_inout[uint2(px, py)] = float4(base + add, 1.0);
        }
    }
}

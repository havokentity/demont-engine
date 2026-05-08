// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// GPU-side auto-exposure pass.
//
// Replaces the per-frame CPU readback of accum_hdr with a tiny
// compute reduction. A single workgroup of 64 threads samples
// accum_hdr at a regular stride, computes the LOG-AVERAGE
// (geometric mean) luminance via group-shared reduction, then
// writes the smoothed exposure scalar back into the persistent
// ExposureState buffer. The path tracer's tonemap reads
// `exposure_state[0]` directly on the next frame -- no CPU readback,
// no queue stall. The engine emits a compute-write -> compute-read
// barrier between the path-trace dispatch and this kernel inside
// the same command buffer (Vulkan needs it explicitly; Metal
// auto-barriers between dispatches on a shared resource).
//
// Why log-average rather than arithmetic mean? Real-world HDR
// scenes produce extreme outliers -- the sun disc reaches ~1.6e9
// nits while the surrounding sky is ~1e4 and ground objects are
// near 1.0. Arithmetic mean of N samples where one pixel is the
// sun and N-1 are scene gives `mean = (1.6e9 + N*scene)/N` ~=
// 1.6e9/N, which dominates regardless of scene content. Auto-
// exposure then computes target = 0.18 / mean ~= 0, gets clamped
// to exp_min, and the actual scene stays burned out because the
// floor prevents real exposure compensation. Log-average uses the
// geometric mean of luminance: extreme outliers are compressed by
// the log so a single sun pixel barely moves the average. This is
// Reinhard's 2002 standard technique and the de-facto base for
// modern game engine auto-exposure.
//
// Switch to a 256-bin histogram if we ever need percentile
// clipping (cinematic look) or HDR display peak picking.
//
// Bindings:
//   binding 1  : accum_hdr (RWTexture2D<float4>) -- shared with PathTrace
//   binding 15 : exposure_state (RWStructuredBuffer<float>, 1 element)
//
// Push constants (32B, fits within the 112B unified push range):
//   width, height: accum_hdr dimensions
//   stride       : sample step in pixels (16 matches CPU code)
//   key          : middle-grey target (r_exposure_target, e.g. 0.18)
//   exp_min/max  : clamps for current_exposure
//   adapt_speed  : log-space lerp factor; <=0 means "lock to target"

// MSL SLOT ALIGNMENT NOTICE
// ==========================
// Slang's MSL backend assigns Metal buffer/texture slots in DECLARATION
// order, ignoring vk::binding numbers. The engine's AutoExposure
// dispatch reuses the bindings PathTrace.slang's dispatch left in
// place (engine texture slot 1 = accum_hdr, engine buffer slot 6 =
// exposure_state). For that to work on Metal, AutoExposure's MSL slot
// layout needs to match PathTrace's:
//
//   PathTrace MSL : tex(0)=output, tex(1)=accum_hdr, ...
//                   buf(0)=AS, buf(1..5)=mesh/cdf storage buffers,
//                   buf(6)=exposure_state, buf(7)=Push
//
// Without alignment, AutoExposure on Metal would emit accum_hdr at
// tex(0) (where the engine has the swapchain bound) and
// exposure_state at buf(0) (unbound), reading garbage and writing to
// nowhere -- exposure_state would never update.
//
// The fix: declare dummies before the actively-used bindings to
// shift them to the right MSL slots, and reference the dummies in a
// runtime-conditional branch the optimizer can't elide so Slang
// actually emits the bindings.
//
// On Vulkan this is a no-op: each declaration uses its vk::binding
// number directly, and the dummies' bindings are already populated
// by the path-trace dispatch's descriptor writes. The dead branch
// gets compiled away by the Vulkan driver's pipeline JIT.

// --- texture slots: padded to put accum_hdr at tex(1) ----------------
[[vk::binding(0, 0)]]  RWTexture2D<float4> _slot_output;
[[vk::binding(1, 0)]]  RWTexture2D<float4> accum_hdr;

// --- buffer slots: AS at buf(0), then 5 storage buffers, then
//     exposure_state at buf(6). Matches PathTrace's emission order.
[[vk::binding(2, 0)]]  RaytracingAccelerationStructure _slot_tlas;
[[vk::binding(3, 0)]]  StructuredBuffer<float>  _slot_mesh_positions;
[[vk::binding(4, 0)]]  StructuredBuffer<uint>   _slot_mesh_indices;
[[vk::binding(5, 0)]]  StructuredBuffer<float4> _slot_primitives;
[[vk::binding(10, 0)]] StructuredBuffer<float>  _slot_marginal_cdf;
[[vk::binding(11, 0)]] StructuredBuffer<float>  _slot_conditional_cdf;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> exposure_state;

[[vk::push_constant]]
cbuffer Push {
    uint  width;
    uint  height;
    uint  stride;
    uint  pad0;
    float key;
    float exp_min;
    float exp_max;
    float adapt_speed;
};

groupshared float g_lum_sum[64];
groupshared uint  g_lum_count[64];

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {
    // SV_GroupIndex (linear, 0..total-1) is the right semantic for the
    // strided reduction loop: every thread maps to a unique position
    // in g_lum_sum / g_lum_count without any 2D->1D math the shader
    // has to do itself. Maps to MSL's [[thread_index_in_threadgroup]]
    // and SPIR-V's LocalInvocationIndex. Always 0..63 here regardless
    // of the underlying threadgroup shape Metal / Vulkan actually
    // dispatched with -- which matters because earlier MetalCommandBuffer
    // versions auto-derived tgsize from the pipeline's threadExecutionWidth
    // (~32 on Apple silicon) and produced a (32,2,1) shape instead of the
    // declared (8,8,1). Today MetalCommandBuffer::Dispatch hardcodes
    // tgsize = (8,8,1) to match every kernel's [numthreads] declaration,
    // so the historical reason is gone, but using SV_GroupIndex stays the
    // right call -- it's still the cleanest, most portable way to write
    // a strided 1D reduction loop.
    // Force Slang to emit the dummy bindings even though they're
    // unused at runtime. The compiler can't statically prove
    // `width == 0xDEADBEEFu` is false (width comes from a runtime
    // push constant), so the branch survives optimization and the
    // referenced declarations get slot assignments. The branch is
    // never taken on real dispatches (width is always a real frame
    // dimension), so this costs one comparison + a never-taken jump
    // per thread -- ~1 cycle.
    if (width == 0xDEADBEEFu) {
        _slot_output[uint2(0, 0)] = float4(0, 0, 0, 0);
        // RaytracingAccelerationStructure can't easily be
        // "touched" inside a compute kernel without ray query --
        // declare it but rely on Slang's reflection emitting it
        // because the binding number is annotated. The buf(0)
        // slot in MSL output is reserved for AS in PathTrace, so
        // this matches even if the AS itself isn't referenced
        // here. (Slang does emit declared-but-unread AS bindings
        // for kernels that don't use them.)
        let p = _slot_mesh_positions[0];
        let i = _slot_mesh_indices[0];
        let q = _slot_primitives[0];
        let m = _slot_marginal_cdf[0];
        let c = _slot_conditional_cdf[0];
        // Touch the real ones too to make the side effect concrete.
        accum_hdr[uint2(0, 0)] = float4(p, float(i), q.x, m + c);
        exposure_state[0] = p;
    }

    uint w_steps = (width  + stride - 1) / stride;
    uint h_steps = (height + stride - 1) / stride;
    uint total   = w_steps * h_steps;

    // Per-thread accumulation in LOG space, with luminance clipped to
    // [1e-6, 1e2] before logging. The lower floor (1e-6) keeps log()
    // finite for near-black pixels; the upper cap (1e2 ~ 100 nits, the
    // sRGB reference-white target) is the critical bit for HDR scenes:
    //
    //   - Real-world sky luminance: ~1e4 nits
    //   - Bright cloud / sun-lit white wall: ~1e3-1e4 nits
    //   - Sun disc: ~1.6e9 nits
    //   - Indoor ambient / shaded scene element: ~0.1-10 nits
    //
    // Without an upper cap, even a small percentage of sky / sun
    // pixels in frame pulls the geometric mean far above the actual
    // mid-tones of the scene. The metering then computes a tiny
    // target exposure (key/avg) that crushes the ground / shaded
    // areas to black while the sky still saturates ACES at the top.
    //
    // The 1e2 cap (~100 nits) means "for metering purposes, anything
    // brighter than ~100 nits is treated as 100 nits". 100 nits is
    // roughly the brightness of a typical SDR display panel and the
    // sRGB spec's reference white -- the natural target for "what
    // the metering should consider fully bright". Anything brighter
    // (sun-lit surfaces ~3e3 nits, sky ~1e4 nits, sun disc ~1.6e9
    // nits) gets the same metering weight as a 100-nit pixel.
    //
    // Scene elements at their actual luminance still flow through
    // the path tracer to the swapchain at full HDR -- only the
    // auto-exposure's "what's the typical scene luminance?"
    // calculation ignores the extreme tail. The bright end of the
    // image still saturates ACES (correct for an SDR display
    // showing midday sun), but mid-tones and shadows can land in
    // the visible range because the metering isn't dragged up by
    // the unbounded HDR tail.
    //
    // The if (lum > 0) guard skips truly zero pixels so they don't
    // pull the mean down toward log(eps) and bias dark-scene exposure.
    float sum  = 0.0;
    uint  cnt  = 0;
    for (uint i = gid; i < total; i += 64u) {
        uint x = (i % w_steps) * stride;
        uint y = (i / w_steps) * stride;
        if (x < width && y < height) {
            float3 c = accum_hdr[uint2(x, y)].rgb;
            float lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
            if (lum > 0.0) {
                sum += log(clamp(lum, 1e-6, 1e2));
                cnt += 1u;
            }
        }
    }
    g_lum_sum[gid]   = sum;
    g_lum_count[gid] = cnt;
    GroupMemoryBarrierWithGroupSync();

    // Single-thread reduction over the 64 partials. The whole job
    // is a few microseconds on a 5090; LDS-tree reduce isn't worth
    // the code complexity here.
    if (gid == 0u) {
        float total_sum   = 0.0;
        uint  total_count = 0u;
        for (uint i = 0u; i < 64u; ++i) {
            total_sum   += g_lum_sum[i];
            total_count += g_lum_count[i];
        }
        if (total_count == 0u) return;

        // Geometric mean: total_sum is the sum of log(lum) values, so
        // exp(total_sum / total_count) is the geometric mean of all
        // sampled luminances. Robust to extreme outliers (sun disc,
        // bright specular highlights) that would dominate an arithmetic
        // mean and force exposure to clamp at exp_min for outdoor
        // scenes regardless of the rest of the frame's luminance.
        float log_avg   = total_sum / float(total_count);
        float avg_lum   = exp(log_avg);
        float target    = key / max(avg_lum, 1e-6);
        target          = clamp(target, exp_min, exp_max);

        float current = exposure_state[0];
        if (current <= 0.0 || adapt_speed <= 0.0) {
            // Either uninitialised, or "locked" preset (DSLR/linear):
            // jump straight to target. Smoothing kicks in only when
            // adapt_speed > 0 AND there's a prior value to fade from.
            current = target;
        } else {
            // Geometric (log-space) lerp: perceptually-uniform fade.
            current = exp(log(current) * (1.0 - adapt_speed) +
                          log(target)  * adapt_speed);
        }
        exposure_state[0] = current;
    }
}

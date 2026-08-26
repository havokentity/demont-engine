// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Acceleration-structure probe kernel (issue #254 P0 / issue #251).
//
// A deliberately minimal ray-query kernel whose ONLY job is to report,
// per ray, what the hardware traversal saw: did it hit, which instance
// did it hit (`CommittedInstanceID()` -- the value the RHI's
// `TLASInstance::instance_id` is contracted to deliver), and how far
// away the hit was. It exists so `tests/rhi_accel_update_test.cpp` can
// assert the multi-instance TLAS contract from the SHADER side rather
// than from host-side bookkeeping, on every backend that can host an
// acceleration structure.
//
// Why a dedicated kernel and not PathTrace.slang: PathTrace reads no
// instance id at all today (that lands with the planetary P4 chunk
// work), so routing a test through it would only assert that the
// megakernel's other 6 000 lines still shade a box. The divergence
// this guards -- Metal silently dropping `instance_id` while Vulkan
// honours it, #251 -- is invisible unless something reads the id back
// and compares it against what the host asked for.
//
// Bindings deliberately reuse indices already present in the Vulkan
// backend's shared descriptor-set layout (2 = acceleration structure,
// 3 = storage buffer) so no layout entry has to be added for a
// test-only kernel. On Metal, Slang assigns its own buffer indices and
// MetalDevice parses them out of the emitted MSL, so the numbers here
// are only meaningful to the SPIR-V target.
[[vk::binding(2, 0)]] RaytracingAccelerationStructure probe_tlas;

// Two uints per ray: [2*i + 0] = 1 if the ray hit anything, else 0.
//                    [2*i + 1] = CommittedInstanceID() of the hit
//                                (undefined-but-zero on a miss).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> probe_out;

[[vk::push_constant]]
cbuffer ProbePush {
    // xyz = world-space origin of ray 0 (metres), w = ray count.
    float4 origin_and_count;
    // xyz = shared world-space ray direction (unit), w = tmax (metres).
    float4 dir_and_tmax;
    // xyz = per-ray origin stride (metres): ray i starts at
    //       origin + stride * i. w = rays per row of the dispatch grid.
    float4 stride_and_row;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    const uint count = uint(origin_and_count.w);
    const uint row   = max(uint(stride_and_row.w), 1u);
    const uint i     = tid.y * row + tid.x;
    if (tid.x >= row || i >= count) return;

    RayDesc r;
    r.Origin    = origin_and_count.xyz + stride_and_row.xyz * float(i);
    r.Direction = dir_and_tmax.xyz;
    r.TMin      = 1e-4;
    r.TMax      = dir_and_tmax.w;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(probe_tlas, RAY_FLAG_NONE, 0xFFu, r);
    while (q.Proceed()) {}

    uint hit = 0u;
    uint id  = 0u;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        hit = 1u;
        // The value under test. Vulkan sources it from
        // VkAccelerationStructureInstanceKHR::instanceCustomIndex;
        // Metal from MTLAccelerationStructureUserIDInstanceDescriptor::
        // userID (which is why the TLAS build has to opt into that
        // descriptor type -- see MetalDevice::CreateTLAS).
        id = q.CommittedInstanceID();
    }
    probe_out[i * 2u + 0u] = hit;
    probe_out[i * 2u + 1u] = id;
}

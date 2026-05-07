// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Compute kernel that fills the bound storage texture with a uniform colour.
// Bound at runtime to slot 0 (texture) and a 16-byte push-constant buffer at
// slot 0 (color). Used by the Metal/Vulkan backends as the P3 smoke test.

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::push_constant]]
cbuffer Push {
    float4 color;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID) {
    uint2 dim;
    output.GetDimensions(dim.x, dim.y);
    if (any(tid >= dim)) return;
    output[tid] = color;
}

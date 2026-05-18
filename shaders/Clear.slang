// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
// Compute kernel that fills the bound storage texture with a uniform colour.
// Bound at runtime to slot 0 (texture) and a 16-byte push-constant buffer at
// slot 0 (color). Used by the Metal/Vulkan backends as the P3 smoke test.

// vk::image_format("rgba8"): this P3 smoke-test kernel writes the
// swapchain (BGRA8_UNORM on both Win/Mac); SPIR-V image format
// rgba8 covers BGRA8 storage-image bindings too (channel order is
// view-side, not type-side). Without the format Slang emits
// Format=Unknown which NVIDIA's compiler can silently no-op the
// store on certain dispatch paths.
[[vk::binding(0, 0), vk::image_format("rgba8")]]
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

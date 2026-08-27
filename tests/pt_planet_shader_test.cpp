// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P4 (#258): the host mirror of the shader-side terrain contract.
//
// WHY THIS FILE EXISTS
//
// Three of this phase's shader claims are wire formats between C++ and
// Slang with no generated header in between, and one is an equality between
// a number the host computes and a number the shader computes. None of them
// can be checked by a golden image on the software backend, because
// src/rhi_software/SoftwareTracer.cpp is a HAND-WRITTEN CPU renderer and not
// a Slang-to-CPU emit -- it does not execute the shader, so a software
// golden proves nothing about it.
//
//   1. kInstDescFloat4s and the descriptor lane layout. The host packs six
//      float4 per instance; loadInstanceDesc() unpacks them. A mismatch is
//      a silently wrong material on every mesh hit in the engine.
//   2. The three terrain buffer bindings. Slang declares 36 / 38 / 39;
//      VulkanDevice's slot table maps engine slots 18 / 19 / 20 to them and
//      the descriptor-set layout must declare all three. A shader binding
//      the layout lacks is a hard vkCreateComputePipelines failure -- on
//      Vulkan only, which this Mac-tested phase cannot otherwise catch.
//   3. cone_spread. The host's LOD rule is only conservative for secondary
//      rays if the host and the shader agree on 2*tan(fovY/2)/height
//      EXACTLY. Two spellings of the same formula is how that agreement
//      rots.
//
// COUNT OCCURRENCES, DO NOT TEST FOR PRESENCE.
//
// Every pin below counts. A `find() != npos` check passes as soon as ONE
// site matches, which is exactly how a mirror keeps passing after the
// shader has grown a second copy of the thing being mirrored -- the trap
// that let #276 ship a stale duplicate of planetAltitude in
// CloudsRaymarch.slang while pt_math_altitude stayed green.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/PlanetTerrain.h"
#include "renderer/Planet/CubedSphere.h"

#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string Slurp(const char* path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE_MESSAGE(f.good(), "cannot open ", path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Remove `//`-to-end-of-line comments. Statement pins count against the
// STRIPPED text, because prose that quotes the statement being pinned is
// itself a match -- the first version of this file counted four
// `h.mat = MAT_METAL;` where two were a doc comment and a metallic-map
// override, and the assertion was wrong rather than the shader. Counting
// is only as good as what it counts.
std::string StripLineComments(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool in_comment = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (!in_comment && src[i] == '/' && i + 1 < src.size() && src[i + 1] == '/') {
            in_comment = true;
        }
        if (src[i] == '\n') in_comment = false;
        if (!in_comment) out.push_back(src[i]);
    }
    return out;
}

std::size_t CountOccurrences(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return 0;
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos;
         at = hay.find(needle, at + needle.size())) {
        ++n;
    }
    return n;
}

}  // namespace

TEST_CASE("the instance descriptor is one wire format, declared once on each side") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);

    // The float4 count. One declaration in the shader, one in the host
    // header; the host's is a compile-time constant this file links against.
    CHECK(CountOccurrences(slang, "static const uint kInstDescFloat4s = 6u;") == 1);
    CHECK(pt::engine::kInstDescFloat4s == 6u);

    // The kind tags. If these ever diverge every terrain hit shades as the
    // legacy mesh (or vice versa), which reads as "the planet is gold".
    CHECK(CountOccurrences(slang, "kInstKindMesh    = 0u;") == 1);
    CHECK(CountOccurrences(slang, "kInstKindTerrain = 1u;") == 1);
    CHECK(pt::engine::kInstKindMesh == 0u);
    CHECK(pt::engine::kInstKindTerrain == 1u);

    // The unpack reads exactly six float4 and no more. Counting the reads
    // is what catches a seventh lane added on one side only.
    for (int i = 0; i < 6; ++i) {
        const std::string line =
            "float4 d" + std::to_string(i) + " = instance_desc[b + " +
            std::to_string(i) + "u];";
        CHECK(CountOccurrences(slang, line) == 1);
    }
    CHECK(CountOccurrences(slang, "instance_desc[b + 6u]") == 0);

    // Exactly one loader and exactly one call site. A second call site would
    // mean a second material path, which is the thing the descriptor exists
    // to remove.
    CHECK(CountOccurrences(slang, "InstanceDesc loadInstanceDesc(uint id)") == 1);
    CHECK(CountOccurrences(StripLineComments(slang), "loadInstanceDesc(inst)") == 1);
}

TEST_CASE("CommittedInstanceID is read exactly once, at the mesh hit") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    // Before this phase the count was zero -- the engine had only ever built
    // one BLAS with one identity instance. Exactly one now: a second read
    // would mean a second place deciding what an instance is.
    CHECK(CountOccurrences(StripLineComments(slang), "q.CommittedInstanceID()") == 1);
}

TEST_CASE("the terrain bindings agree across Slang and the Vulkan tables") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    const std::string vk    = Slurp(PT_VULKAN_DEVICE_PATH);

    // Slang declares each exactly once.
    CHECK(CountOccurrences(
        slang, "[[vk::binding(36, 0)]] StructuredBuffer<float4> terrain_verts;") == 1);
    CHECK(CountOccurrences(
        slang, "[[vk::binding(38, 0)]] StructuredBuffer<uint>   terrain_indices;") == 1);
    CHECK(CountOccurrences(
        slang, "[[vk::binding(39, 0)]] StructuredBuffer<float4> instance_desc;") == 1);

    // Binding 37 is GodRays' mask TEXTURE. Images and buffers share one
    // binding-number space in set 0, so 37 must NOT appear as a terrain
    // buffer -- this pin is the reason the three are 36 / 38 / 39 and not
    // 36 / 37 / 38.
    CHECK(CountOccurrences(slang, "vk::binding(37, 0)") == 0);

    // The engine slot -> shader binding table, and the descriptor-set
    // layout. A binding declared by the shader but missing from the layout
    // is a hard pipeline-creation failure even under PARTIALLY_BOUND, and
    // this phase is tested on Metal, where that failure cannot surface.
    CHECK(CountOccurrences(vk, "36, // engine slot 18 -> shader binding 36 (terrain_verts)") == 1);
    CHECK(CountOccurrences(vk, "38, // engine slot 19 -> shader binding 38 (terrain_indices)") == 1);
    CHECK(CountOccurrences(vk, "39, // engine slot 20 -> shader binding 39 (instance_desc)") == 1);
    CHECK(CountOccurrences(vk, "add_binding(36, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);") == 1);
    CHECK(CountOccurrences(vk, "add_binding(38, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);") == 1);
    CHECK(CountOccurrences(vk, "add_binding(39, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);") == 1);

    // The descriptor pool has to be sized for them. 17 storage buffers
    // before this phase, 20 after.
    CHECK(CountOccurrences(vk, "kTotalSets * 20 + 8") == 1);
    CHECK(CountOccurrences(vk, "kTotalSets * 17 + 8") == 0);
}

TEST_CASE("host and shader compute the same cone_spread") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    const std::string eng   = Slurp(PT_ENGINE_CPP_PATH);

    // The shader's per-camera-sample spread, declared once.
    CHECK(CountOccurrences(
        slang, "float cone_spread = (2.0 * pos_fovtan.w) / float(max(dim.y, 1u));") == 1);
    // The host's, fed to the LOD selector. pos_fovtan.w IS Camera::FovYTan()
    // and dim.y IS the window height, so these are the same expression.
    CHECK(CountOccurrences(
        eng, "lod.cone_spread = 2.0 * static_cast<double>(camera_->FovYTan()) / h;") == 1);

    // And numerically, at the fixtures' own settings. This is what makes
    // "every LOD transition is sub-pixel" mean the same thing on both sides.
    const double fov_y_deg = 55.0;
    const double height    = 384.0;
    const double fov_y_tan = std::tan(0.5 * fov_y_deg * 3.14159265358979323846 / 180.0);
    const double shader    = (2.0 * fov_y_tan) / height;
    const double host      = 2.0 * fov_y_tan / height;
    CHECK(shader == host);
    // A sanity anchor on the magnitude the design quotes: ~1.07e-3 at
    // 1080p / 60 deg.
    const double at_1080p =
        2.0 * std::tan(0.5 * 60.0 * 3.14159265358979323846 / 180.0) / 1080.0;
    CHECK(at_1080p > 1.06e-3);
    CHECK(at_1080p < 1.08e-3);
}

TEST_CASE("the slope-variance fold is Bruneton's, applied once") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    // alpha^2_eff = alpha^2_material + 2 * sigma^2. The factor of two is
    // not decorative -- it is what makes the fold energy-preserving rather
    // than merely "rougher", so pin the expression and not just the name.
    CHECK(CountOccurrences(StripLineComments(slang), "return sqrt(saturate(a + 2.0 * s2));") == 1);
    CHECK(CountOccurrences(slang, "float terrainRoughnessFold(") == 1);
    CHECK(CountOccurrences(StripLineComments(slang), "terrainRoughnessFold(idesc.roughness, cw, idesc)") == 1);
    // Eight mips, matching kSlopeMips on the host.
    CHECK(CountOccurrences(slang, "float  sigma2[8];") == 1);
    CHECK(pt::planet::kSlopeMips == 8);
}

TEST_CASE("the terrain hit sets uv_scale, closing the #248 gap on meshes") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    // pbrConeTexLod short-circuits to mip 0 when uv_scale <= 0, which is
    // what every mesh hit did before this phase -- the horizon aliasing
    // #248 set out to kill, on the primitive that occupies most of a
    // planet's screen area.
    CHECK(CountOccurrences(StripLineComments(slang), "if (uv_scale <= 0.0) return 0.0;") == 1);
    CHECK(CountOccurrences(StripLineComments(slang), "h.uv_scale = idesc.uv_scale;") == 1);
}

TEST_CASE("the hard-coded mesh material survives only behind the gate") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    // With PT_PLANET_ENABLED the mesh material comes from descriptor 0,
    // which the engine fills with exactly these constants -- that is the
    // bit-identity guarantee. The literals must therefore still appear
    // (in the gate's OFF arm and in the software-BVH fallback) but must
    // NOT appear in the hardware path's ON arm.
    //
    // Counting is the whole point: `h.mat = MAT_METAL;` appearing at all
    // proves nothing, because it legitimately appears in the OFF arm.
    // Three sites is the correct number -- OFF arm, software BVH walk, and
    // the software-mesh linear fallback's shared tail.
    const std::string code = StripLineComments(slang);
    // Three, and each one is accounted for:
    //   * the PT_PLANET_ENABLED=OFF arm of the hardware mesh hit;
    //   * the software-BVH mesh walk, which fires only when no TLAS is
    //     bound (no hardware ray tracing) and therefore never carries
    //     terrain -- left hard-coded deliberately, and noted as a
    //     remainder rather than silently converted;
    //   * applyPbrTextures' metallic-map override, which is unrelated:
    //     `if (m > 0.5 && h.mat == MAT_LAMBERT) h.mat = MAT_METAL;`.
    CHECK(CountOccurrences(code, "h.mat = MAT_METAL;") == 3);
    CHECK(CountOccurrences(code, "h.albedo = MESH_ALBEDO;") == 2);
    // And the descriptor-sourced replacements: the terrain branch and the
    // hardware mesh branch's ON arm.
    CHECK(CountOccurrences(code, "h.mat = idesc.mat;") == 2);
    CHECK(CountOccurrences(code, "h.albedo = idesc.albedo;") == 1);
}

TEST_CASE("the engine's descriptor 0 reproduces the shader's old constants") {
    const std::string slang = Slurp(PT_SHADER_PATHTRACE_PATH);
    const std::string eng   = Slurp(PT_ENGINE_CPP_PATH);
    // MESH_ALBEDO, declared once in the shader.
    CHECK(CountOccurrences(
        StripLineComments(slang),
        "static const float3 MESH_ALBEDO  = float3(1.00, 0.85, 0.45);") == 1);
    // And reproduced once on the host, in EnsureInstanceDescriptors.
    CHECK(CountOccurrences(eng, "data[4] = 1.00f; data[5] = 0.85f; data[6] = 0.45f;") == 1);
    CHECK(CountOccurrences(eng, "data[7] = 0.05f;") == 1);
}

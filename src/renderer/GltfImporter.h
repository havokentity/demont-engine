// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

// glTF 2.0 importer -- MVP scope (#79).
//
// Parses a single static mesh out of a `.gltf` or `.glb` and returns
// positions/indices/normals in exactly the shape pt::csg::BakedMesh
// produces, so the Engine can hand the result straight to
// RebuildMeshResources() without any per-vertex repacking.
//
// MVP scope (intentionally narrow -- the parent issue lists the full
// long-term plan, but this PR only ships the static-mesh slice):
//   - One mesh, one primitive. The first mesh in the file referenced
//     by the default scene is the one we import; if the file has no
//     scene root we fall back to data->meshes[0].
//   - Triangles only. Other primitive topologies (lines, points, strips,
//     fans) are rejected with a clean error.
//   - Positions are required. Normals are imported if present and
//     synthesized per-face if absent (we still want shading-correct
//     output for hand-modeled meshes that omit them, e.g. Box.gltf
//     ships normals, but plenty of converters don't).
//   - One root-node world transform is baked into the positions /
//     normals at import time. We do not preserve the node hierarchy
//     -- the engine's mesh pipeline (CSG / triangle BVH) consumes a
//     single triangle soup. Full scene-graph import is deferred.
//   - Base-color factor and base-color texture URI are recorded on
//     the result for the consumer; ACTUAL texture upload + material
//     plumbing (the per-mesh shading uniform that the path tracer
//     uses) is owned by the textured-materials work (#74) and is
//     intentionally NOT wired here.
//
// Explicitly DEFERRED to follow-ups (do not let scope creep in):
//   - Multi-primitive meshes, multi-mesh scenes, node hierarchies.
//   - Skeletons, skinning, morph targets, animations.
//   - PBR metalness / roughness / normal / occlusion / emissive maps.
//   - KHR_lights_punctual, cameras, KTX2 / Draco / EXT_meshopt_compression.
//   - Tangents (only needed when normal mapping lands with #74).
//
// Coordinate system: glTF 2.0 uses right-handed +Y up, -Z forward, the
// same convention our existing CSG / scene math uses. No axis flip.
//
// Units: glTF specifies metres. The engine memory says 1 world unit =
// 1 metre, so we ingest verbatim with no scaling.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pt::renderer {

// A parsed glTF base-color texture (deferred plumbing -- see header).
// width/height in pixels; pixels is tightly packed RGBA8 in sRGB-encoded
// (nonlinear) space as glTF requires -- the spec says baseColorTexture
// pixels are sRGB-encoded, so the consumer must decode to linear before
// lighting. Empty pixels means "no texture, factor only".
struct GltfBaseColorTexture {
    std::uint32_t            width  = 0;
    std::uint32_t            height = 0;
    std::vector<std::uint8_t> pixels;   // 4 bytes per texel (RGBA8 sRGB)

    bool Empty() const { return pixels.empty() || width == 0 || height == 0; }
};

// Result of a successful import. The geometry layout is byte-identical
// to pt::csg::BakedMesh (positions/normals = xyz-packed floats, indices
// = 3 per triangle, CCW from outside per glTF convention) so the engine
// can hand it straight to RebuildMeshResources.
struct GltfMesh {
    std::vector<float>         positions;   // 3 floats per vertex
    std::vector<float>         normals;     // 3 floats per vertex
    std::vector<std::uint32_t> indices;     // 3 per triangle
    // Wave 8 PBR (#26): per-vertex texture coordinates from TEXCOORD_0.
    // 2 floats (u, v) per vertex, parallel to positions. Empty when the
    // primitive has no TEXCOORD_0 attribute (the engine then renders the
    // mesh with its flat material -- no texture sampling). glTF UVs use
    // the top-left origin convention (v=0 at the top of the image), the
    // same as the engine's texel-fetch atlas sampling.
    std::vector<float>         uvs;

    // Material -- base color only (MVP). RGBA in linear space (glTF's
    // pbrMetallicRoughness.baseColorFactor is already linear per spec).
    // Default is the glTF default {1, 1, 1, 1} when the file has no
    // material on the primitive.
    float                      base_color_factor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GltfBaseColorTexture       base_color_texture;   // Empty() if absent.

    // Provenance for logs / diagnostics. Mesh name from the glTF
    // (data->meshes[*].name) -- empty if the file didn't name the mesh.
    std::string                source_path;
    std::string                mesh_name;

    std::uint32_t VertexCount()   const { return static_cast<std::uint32_t>(positions.size() / 3); }
    std::uint32_t TriangleCount() const { return static_cast<std::uint32_t>(indices.size() / 3); }
    bool          Empty()         const { return indices.empty(); }
};

// Parse a .gltf or .glb at `path`. Returns std::nullopt on failure;
// `out_error` (optional) receives a short human-readable diagnostic
// suitable for echoing to the console. On success the returned mesh
// has at least one triangle and a normal per vertex (synthesized
// per-face if the file omitted them).
std::optional<GltfMesh> LoadGltf(const std::string& path,
                                  std::string*       out_error = nullptr);

}  // namespace pt::renderer

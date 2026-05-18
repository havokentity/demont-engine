// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "GltfImporter.h"

#include "../core/Log.h"

// cgltf is the upstream single-header glTF 2.0 parser (MIT, jkuhlmann).
// Vendored at third_party/cgltf/cgltf.h, see issue #79 for rationale
// (single-header + no transitive deps + actively-maintained + the
// idiomatic choice in the Khronos ecosystem). The implementation is
// defined in the sibling gltf_impl.cpp TU -- it is the SOLE definer
// of CGLTF_IMPLEMENTATION in the codebase. If a future TU needs the
// parser, it should #include "GltfImporter.h" and call LoadGltf, not
// redefine the impl.
//
// Same one-TU-defines-impl pattern applies to stb_image (also in
// gltf_impl.cpp). The rest of the engine uses stb_image_write
// (separate symbol table) from src/engine/stb_impl.cpp; we add
// stb_image's read path in gltf_impl.cpp only to decode the optional
// baseColorTexture.
//
// CMakeLists.txt scopes a per-file warning suppression to just
// gltf_impl.cpp -- this TU (project logic) keeps strict warnings.
#include "../../third_party/cgltf/cgltf.h"
#include "../../third_party/stb/stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>

namespace pt::renderer {

namespace {

// RAII wrapper around cgltf_data*; cgltf_free is a no-op on nullptr
// per the upstream contract, so destruction is safe in any path.
struct CgltfDataDeleter {
    void operator()(cgltf_data* d) const noexcept {
        if (d) cgltf_free(d);
    }
};
using CgltfDataPtr = std::unique_ptr<cgltf_data, CgltfDataDeleter>;

// cgltf_result -> short human-readable diagnostic. Mirrors the upstream
// enum spelling for greppability.
const char* CgltfResultName(cgltf_result r) {
    switch (r) {
        case cgltf_result_success:           return "success";
        case cgltf_result_data_too_short:    return "data too short";
        case cgltf_result_unknown_format:    return "unknown format";
        case cgltf_result_invalid_json:      return "invalid JSON";
        case cgltf_result_invalid_gltf:      return "invalid glTF";
        case cgltf_result_invalid_options:   return "invalid options";
        case cgltf_result_file_not_found:    return "file not found";
        case cgltf_result_io_error:          return "I/O error";
        case cgltf_result_out_of_memory:     return "out of memory";
        case cgltf_result_legacy_gltf:       return "legacy glTF (1.0 not supported)";
        default:                             return "unknown error";
    }
}

// Pick the mesh we want to import. Strategy:
//   1. If the file has a default scene (data->scene), walk it
//      breadth-first and return the first node whose mesh is non-null.
//   2. Otherwise, if scenes exist, scan scenes[0] the same way.
//   3. Otherwise, fall back to data->meshes[0] with the identity
//      world transform.
//
// Returns nullptr in `out_node` when the fallback path fires (the
// returned mesh exists but isn't reached via a node, so there's no
// world transform to bake -- treat it as identity).
const cgltf_mesh* SelectMesh(const cgltf_data* data, const cgltf_node** out_node) {
    *out_node = nullptr;
    if (!data) return nullptr;

    // Search a single scene (depth-first via the nodes array). cgltf
    // already linearized the hierarchy for us via scene->nodes; each
    // root is walked transitively through its children.
    auto search_scene = [&](const cgltf_scene* scene) -> const cgltf_mesh* {
        if (!scene) return nullptr;
        // BFS to keep behaviour predictable across DCCs (Blender, Maya,
        // Substance all emit different orderings, but root-first BFS
        // makes "first mesh in scene" mean what the user expects).
        std::vector<const cgltf_node*> queue;
        queue.reserve(scene->nodes_count + 16);
        for (cgltf_size i = 0; i < scene->nodes_count; ++i) {
            if (scene->nodes[i]) queue.push_back(scene->nodes[i]);
        }
        for (cgltf_size head = 0; head < queue.size(); ++head) {
            const cgltf_node* n = queue[head];
            if (n->mesh) {
                *out_node = n;
                return n->mesh;
            }
            for (cgltf_size i = 0; i < n->children_count; ++i) {
                if (n->children[i]) queue.push_back(n->children[i]);
            }
        }
        return nullptr;
    };

    if (data->scene) {
        if (auto* m = search_scene(data->scene)) return m;
    }
    if (data->scenes_count > 0) {
        if (auto* m = search_scene(&data->scenes[0])) return m;
    }
    if (data->meshes_count > 0) {
        return &data->meshes[0];
    }
    return nullptr;
}

// Apply a 4x4 column-major glTF world transform to a 3-component
// position OR direction (controlled by `is_position`). Column-major
// because cgltf_node_transform_world fills the output the same way --
// see comment block atop cgltf.h around node transforms.
//
// For positions: full 4x4 with w=1. For normals: 3x3 inverse-transpose
// (we approximate with the upper-left 3x3 since glTF transforms are
// typically rigid for the MVP; we re-normalize after to absorb any
// non-uniform-scale error). A full inverse-transpose would be needed
// for non-uniform scale; that's a deferred follow-up.
void TransformVec3(const float m[16], bool is_position, float v[3]) {
    const float x = v[0], y = v[1], z = v[2];
    if (is_position) {
        v[0] = m[0]*x + m[4]*y + m[8] *z + m[12];
        v[1] = m[1]*x + m[5]*y + m[9] *z + m[13];
        v[2] = m[2]*x + m[6]*y + m[10]*z + m[14];
    } else {
        v[0] = m[0]*x + m[4]*y + m[8] *z;
        v[1] = m[1]*x + m[5]*y + m[9] *z;
        v[2] = m[2]*x + m[6]*y + m[10]*z;
        const float len2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        if (len2 > 0.0f) {
            const float inv = 1.0f / std::sqrt(len2);
            v[0] *= inv; v[1] *= inv; v[2] *= inv;
        }
    }
}

// Generate per-face (flat) geometric normals when the primitive doesn't
// supply them. Writes 3 floats per *vertex*; each vertex shared by N
// triangles gets the average (renormalized) face normal. This matches
// the BakedMesh contract (one normal per vertex) and is good enough for
// path tracing of files like Box.gltf that omit normals.
void SynthesizeNormals(const std::vector<float>&         positions,
                       const std::vector<std::uint32_t>& indices,
                       std::vector<float>&               normals) {
    const std::size_t vert_count = positions.size() / 3;
    normals.assign(vert_count * 3, 0.0f);

    for (std::size_t t = 0; t + 2 < indices.size(); t += 3) {
        const std::uint32_t i0 = indices[t + 0];
        const std::uint32_t i1 = indices[t + 1];
        const std::uint32_t i2 = indices[t + 2];
        const float* p0 = &positions[i0 * 3];
        const float* p1 = &positions[i1 * 3];
        const float* p2 = &positions[i2 * 3];
        const float e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
        const float e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
        const float n[3]  = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        };
        for (std::uint32_t idx : { i0, i1, i2 }) {
            float* dst = &normals[idx * 3];
            dst[0] += n[0]; dst[1] += n[1]; dst[2] += n[2];
        }
    }

    for (std::size_t v = 0; v < vert_count; ++v) {
        float* n = &normals[v * 3];
        const float len2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
        if (len2 > 0.0f) {
            const float inv = 1.0f / std::sqrt(len2);
            n[0] *= inv; n[1] *= inv; n[2] *= inv;
        } else {
            // Degenerate fan -- emit a sentinel +Y normal rather than 0
            // (a zero normal would NaN the path tracer's BRDF cos terms).
            n[0] = 0.0f; n[1] = 1.0f; n[2] = 0.0f;
        }
    }
}

// Resolve a cgltf_image* to RGBA8 pixel data. Handles three forms:
//   1. Embedded data URI ("data:..."): cgltf_load_buffers handled the
//      base64 decode; we read from image->buffer_view if set, else
//      from the image data the loader allocated.
//   2. GLB-embedded buffer view (no URI, buffer_view set).
//   3. External file (URI is a relative filesystem path).
//
// Output `out` is set to RGBA8 (4 bytes per texel) in row-major top-left
// origin -- matches stb_image's default. Returns false on any failure
// (out is reset to Empty()).
bool LoadGltfImage(const cgltf_data*    data,
                   const cgltf_image*   img,
                   const std::string&   gltf_dir,
                   GltfBaseColorTexture& out,
                   std::string&         err) {
    out = GltfBaseColorTexture{};
    if (!img) return false;

    int w = 0, h = 0, comp = 0;
    unsigned char* pixels = nullptr;

    if (img->buffer_view) {
        const cgltf_buffer_view* bv  = img->buffer_view;
        const cgltf_buffer*      buf = bv->buffer;
        if (!buf || !buf->data) { err = "image buffer_view has no loaded buffer"; return false; }
        const unsigned char* base = static_cast<const unsigned char*>(buf->data) + bv->offset;
        pixels = stbi_load_from_memory(base, static_cast<int>(bv->size),
                                       &w, &h, &comp, /*req_comp=*/4);
        if (!pixels) {
            err = std::string("stb_image decode (embedded) failed: ") + stbi_failure_reason();
            return false;
        }
    } else if (img->uri && img->uri[0] != '\0') {
        // Either a data: URI or a relative filesystem path. cgltf's
        // own helpers don't decode the data URI for us (cgltf_load_buffers
        // does that for *buffer* URIs, but image URIs go via cgltf_data's
        // own data: handler only when buffer_view is present). For the
        // MVP scope we accept relative paths only and skip embedded
        // base64 in image URIs -- in practice the canonical Box.gltf
        // sample uses external buffers + external images, and .glb
        // embeds via buffer_view (handled above).
        std::string uri = img->uri;
        if (uri.rfind("data:", 0) == 0) {
            err = "embedded data: URI image not supported in MVP -- use .glb or external file";
            return false;
        }
        // glTF 2.0 spec mandates URI strings be UTF-8 (and either percent-
        // encoded or a data: URI). std::filesystem::u8path was the historic
        // "string is UTF-8" path constructor, but it's deprecated in C++20
        // (libc++ flags -Wdeprecated-declarations on Xcode 26+, breaking
        // our zero-warning CI gate). The non-deprecated successor takes a
        // char8_t range, but introducing char8_t at this single call site
        // would force a u8string round-trip we don't otherwise need.
        // Constructing std::filesystem::path directly from std::string
        // matches the existing codebase pattern (see line 427 below and
        // tools/pt_render_one_frame/main.cpp) and is UTF-8-correct on
        // every platform we ship to (POSIX + Apple use UTF-8 natively for
        // path::native_t == std::string; on Windows MSVC path(std::string)
        // applies the active code page -- if/when we add Windows builds
        // we'll revisit with a project-wide u8string helper, tracked
        // alongside any future i18n work).
        const std::filesystem::path candidate =
            std::filesystem::path(gltf_dir) / std::filesystem::path(uri);
        pixels = stbi_load(candidate.string().c_str(), &w, &h, &comp, /*req_comp=*/4);
        if (!pixels) {
            err = std::string("stb_image load '") + candidate.string() + "' failed: " +
                  stbi_failure_reason();
            return false;
        }
    } else {
        err = "image has neither buffer_view nor URI";
        return false;
    }

    if (w <= 0 || h <= 0) {
        stbi_image_free(pixels);
        err = "decoded image has non-positive extent";
        return false;
    }

    out.width  = static_cast<std::uint32_t>(w);
    out.height = static_cast<std::uint32_t>(h);
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u;
    out.pixels.assign(pixels, pixels + bytes);
    stbi_image_free(pixels);
    (void)data;  // reserved for future cross-buffer validation
    return true;
}

}  // anonymous namespace

std::optional<GltfMesh> LoadGltf(const std::string& path, std::string* out_error) {
    auto fail = [&](std::string msg) -> std::optional<GltfMesh> {
        if (out_error) *out_error = msg;
        LOG_WARN("glTF '{}': {}", path, msg);
        return std::nullopt;
    };

    if (path.empty()) return fail("empty path");
    if (!std::filesystem::exists(path)) {
        return fail("file not found");
    }

    // cgltf_parse_file decides .gltf vs .glb internally by sniffing the
    // first bytes (the 'glTF' magic for binary) so we don't need to
    // dispatch on extension.
    cgltf_options options{};   // zero-init -> default heap allocators + auto type.
    cgltf_data*   raw = nullptr;
    cgltf_result  r   = cgltf_parse_file(&options, path.c_str(), &raw);
    CgltfDataPtr data(raw);
    if (r != cgltf_result_success) {
        return fail(std::string("parse failed: ") + CgltfResultName(r));
    }

    // Load external .bin buffers + decode embedded base64 buffers. This
    // is what fills cgltf_buffer::data so accessor reads work.
    r = cgltf_load_buffers(&options, data.get(), path.c_str());
    if (r != cgltf_result_success) {
        return fail(std::string("buffer load failed: ") + CgltfResultName(r));
    }

    r = cgltf_validate(data.get());
    if (r != cgltf_result_success) {
        return fail(std::string("validation failed: ") + CgltfResultName(r));
    }

    const cgltf_node* root_node = nullptr;
    const cgltf_mesh* mesh = SelectMesh(data.get(), &root_node);
    if (!mesh) return fail("file contains no mesh");

    if (mesh->primitives_count == 0) return fail("mesh has no primitives");
    if (mesh->primitives_count > 1) {
        // MVP: take the first primitive. Log so users notice meshes that
        // were partially imported -- a Sponza-style multi-material mesh
        // would only render its first slice.
        LOG_INFO("glTF '{}': mesh has {} primitives, importing primitive[0] only (MVP)",
                 path, mesh->primitives_count);
    }
    const cgltf_primitive& prim = mesh->primitives[0];

    if (prim.type != cgltf_primitive_type_triangles) {
        return fail("primitive is not a triangle list (lines/points/strips/fans deferred)");
    }
    if (!prim.indices) {
        // The spec allows non-indexed primitives (draw arrays). The
        // engine's BVH builder and SW mesh path both consume indexed
        // triangle soup, so synthesize a trivial 0..N-1 index list
        // when the file ships unindexed.  Handled below after we read
        // the position accessor.
    }

    // --- Positions (REQUIRED) ----------------------------------------
    const cgltf_accessor* pos_acc = nullptr;
    const cgltf_accessor* nrm_acc = nullptr;
    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.type == cgltf_attribute_type_position && attr.index == 0) {
            pos_acc = attr.data;
        } else if (attr.type == cgltf_attribute_type_normal && attr.index == 0) {
            nrm_acc = attr.data;
        }
    }
    if (!pos_acc) return fail("primitive has no POSITION attribute");
    if (pos_acc->count == 0) return fail("POSITION accessor is empty");

    GltfMesh out;
    out.source_path = path;
    out.mesh_name   = mesh->name ? mesh->name : "";
    out.positions.resize(pos_acc->count * 3, 0.0f);
    for (cgltf_size v = 0; v < pos_acc->count; ++v) {
        if (!cgltf_accessor_read_float(pos_acc, v, &out.positions[v * 3], 3)) {
            return fail("POSITION read failed");
        }
    }

    // --- Indices ------------------------------------------------------
    if (prim.indices) {
        if (prim.indices->count % 3 != 0) {
            return fail("index count is not a multiple of 3");
        }
        out.indices.resize(prim.indices->count, 0u);
        for (cgltf_size i = 0; i < prim.indices->count; ++i) {
            cgltf_uint v = 0;
            if (!cgltf_accessor_read_uint(prim.indices, i, &v, 1)) {
                return fail("INDEX read failed");
            }
            out.indices[i] = static_cast<std::uint32_t>(v);
        }
    } else {
        // Draw-arrays primitive: synthesize the implicit index list.
        if (pos_acc->count % 3 != 0) {
            return fail("non-indexed primitive: vertex count is not a multiple of 3");
        }
        out.indices.resize(pos_acc->count);
        for (cgltf_size i = 0; i < pos_acc->count; ++i) {
            out.indices[i] = static_cast<std::uint32_t>(i);
        }
    }

    // --- Normals ------------------------------------------------------
    if (nrm_acc && nrm_acc->count == pos_acc->count) {
        out.normals.resize(nrm_acc->count * 3, 0.0f);
        for (cgltf_size v = 0; v < nrm_acc->count; ++v) {
            if (!cgltf_accessor_read_float(nrm_acc, v, &out.normals[v * 3], 3)) {
                return fail("NORMAL read failed");
            }
        }
    } else {
        if (nrm_acc) {
            LOG_INFO("glTF '{}': NORMAL accessor count {} != POSITION count {}, synthesizing",
                     path, nrm_acc->count, pos_acc->count);
        }
        SynthesizeNormals(out.positions, out.indices, out.normals);
    }

    // --- World transform from root node ------------------------------
    // If the mesh was reached via a node, fold that node's full world
    // transform into the geometry so the engine doesn't need to track
    // per-mesh transforms (MVP). Identity if the mesh was the bare
    // fallback (meshes[0] without any node referencing it).
    if (root_node) {
        float m[16];
        cgltf_node_transform_world(root_node, m);
        // Check for non-identity before iterating (saves work on the
        // common Box.gltf case where the canonical orientation is the
        // identity-rotation default scene).
        constexpr float kIdentity[16] = {
            1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1
        };
        bool is_identity = true;
        for (int i = 0; i < 16; ++i) {
            if (std::fabs(m[i] - kIdentity[i]) > 1e-6f) { is_identity = false; break; }
        }
        if (!is_identity) {
            const std::size_t vc = out.positions.size() / 3;
            for (std::size_t v = 0; v < vc; ++v) {
                TransformVec3(m, /*is_position=*/true,  &out.positions[v * 3]);
                TransformVec3(m, /*is_position=*/false, &out.normals  [v * 3]);
            }
        }
    }

    // --- Material (base-color factor + optional texture) -------------
    if (prim.material && prim.material->has_pbr_metallic_roughness) {
        const auto& pbr = prim.material->pbr_metallic_roughness;
        out.base_color_factor[0] = pbr.base_color_factor[0];
        out.base_color_factor[1] = pbr.base_color_factor[1];
        out.base_color_factor[2] = pbr.base_color_factor[2];
        out.base_color_factor[3] = pbr.base_color_factor[3];
        if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image) {
            const std::string gltf_dir =
                std::filesystem::path(path).parent_path().string();
            std::string img_err;
            if (!LoadGltfImage(data.get(),
                               pbr.base_color_texture.texture->image,
                               gltf_dir,
                               out.base_color_texture,
                               img_err)) {
                // Texture load failure is non-fatal -- we still have
                // the base color factor and the geometry. Log + continue.
                LOG_WARN("glTF '{}': base color texture load failed: {}", path, img_err);
                out.base_color_texture = GltfBaseColorTexture{};
            }
        }
    }

    if (out.Empty()) return fail("imported mesh has zero triangles");

    LOG_INFO("glTF '{}' loaded: mesh='{}', {} verts, {} tris, base color ({:.3f} {:.3f} {:.3f} {:.3f}), tex {}",
             path,
             out.mesh_name,
             out.VertexCount(),
             out.TriangleCount(),
             out.base_color_factor[0], out.base_color_factor[1],
             out.base_color_factor[2], out.base_color_factor[3],
             out.base_color_texture.Empty() ? "absent" : "present");

    return out;
}

}  // namespace pt::renderer

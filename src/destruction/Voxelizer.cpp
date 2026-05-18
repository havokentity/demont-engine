// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Voxelizer.h"

#include "../core/Log.h"
#include "../renderer/AnalyticBvh.h"
#include "../renderer/Csg/CsgScene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace pt::destruction {

namespace {

// -----------------------------------------------------------------------------
// Host-side SDF leaf-shape distance evaluation. Mirrors the formulas
// in shaders/SdfPrimitives.slang (sdSphere / sdBox / sdRoundBox /
// sdTorus / sdCapsule / sdPlane). Kept in lockstep with the GPU side
// so a voxel marked "inside" here matches what the path tracer
// would render at the same point.
// -----------------------------------------------------------------------------
inline float SdSphere(float px, float py, float pz, float r) {
    return std::sqrt(px*px + py*py + pz*pz) - r;
}

inline float SdBox(float px, float py, float pz, float bx, float by, float bz) {
    const float qx = std::fabs(px) - bx;
    const float qy = std::fabs(py) - by;
    const float qz = std::fabs(pz) - bz;
    const float mx = std::max(qx, 0.0f);
    const float my = std::max(qy, 0.0f);
    const float mz = std::max(qz, 0.0f);
    const float outside = std::sqrt(mx*mx + my*my + mz*mz);
    const float inside  = std::min(std::max(qx, std::max(qy, qz)), 0.0f);
    return outside + inside;
}

inline float SdRoundBox(float px, float py, float pz, float bx, float by, float bz, float r) {
    return SdBox(px, py, pz, bx, by, bz) - r;
}

inline float SdTorus(float px, float py, float pz, float R, float r) {
    const float qx = std::sqrt(px*px + pz*pz) - R;
    const float qy = py;
    return std::sqrt(qx*qx + qy*qy) - r;
}

inline float SdCapsule(float px, float py, float pz, float h, float r) {
    const float y  = std::clamp(py, -h, h);
    const float dy = py - y;
    return std::sqrt(px*px + dy*dy + pz*pz) - r;
}

inline float SdPlane(float px, float py, float pz,
                     float nx, float ny, float nz, float d) {
    return px*nx + py*ny + pz*nz + d;
}

inline float OpSmoothUnion(float d1, float d2, float k) {
    if (k <= 0.0f) return std::min(d1, d2);
    const float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return d2 + (d1 - d2) * h - k * h * (1.0f - h);
}

inline float OpSmoothSubtract(float d1, float d2, float k) {
    if (k <= 0.0f) return std::max(d1, -d2);
    const float h = std::clamp(0.5f - 0.5f * (d2 + d1) / k, 0.0f, 1.0f);
    return d1 + (-d2 - d1) * h + k * h * (1.0f - h);
}

inline float OpSmoothIntersect(float d1, float d2, float k) {
    if (k <= 0.0f) return std::max(d1, d2);
    const float h = std::clamp(0.5f - 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return d2 + (d1 - d2) * h + k * h * (1.0f - h);
}

inline float OpDisplace(float d, float amp) {
    return d + amp;
}

float EvalLeafDist(float px, float py, float pz,
                   std::uint32_t shape, const float params[4]) {
    using pt::renderer::SDF_SHAPE_SPHERE;
    using pt::renderer::SDF_SHAPE_BOX;
    using pt::renderer::SDF_SHAPE_ROUNDED_BOX;
    using pt::renderer::SDF_SHAPE_TORUS;
    using pt::renderer::SDF_SHAPE_CAPSULE;
    using pt::renderer::SDF_SHAPE_PLANE;
    switch (shape) {
        case SDF_SHAPE_SPHERE:      return SdSphere(px, py, pz, params[0]);
        case SDF_SHAPE_BOX:         return SdBox(px, py, pz, params[0], params[1], params[2]);
        case SDF_SHAPE_ROUNDED_BOX: return SdRoundBox(px, py, pz, params[0], params[1], params[2], params[3]);
        case SDF_SHAPE_TORUS:       return SdTorus(px, py, pz, params[0], params[1]);
        case SDF_SHAPE_CAPSULE:     return SdCapsule(px, py, pz, params[0], params[1]);
        case SDF_SHAPE_PLANE:       return SdPlane(px, py, pz, params[0], params[1], params[2], params[3]);
        default:                    return 1e30f;
    }
}

// Walk the SdfPrim's flat node array in ascending order (the host
// always emits children before parents -- see AnalyticBvh.h SdfNode
// comment) and return the distance at the root (= last node).
float SdfClusterDist(const pt::renderer::SdfPrim& c, float px, float py, float pz) {
    if (c.node_count == 0u) return 1e30f;
    std::array<float, pt::renderer::SdfPrim::kMaxNodes> dists{};
    for (std::uint32_t i = 0; i < c.node_count && i < pt::renderer::SdfPrim::kMaxNodes; ++i) {
        const pt::renderer::SdfNode& n = c.nodes[i];
        float d;
        const float lx = px - n.center[0];
        const float ly = py - n.center[1];
        const float lz = pz - n.center[2];
        switch (n.op) {
            case pt::renderer::SDF_OP_LEAF:
                d = EvalLeafDist(lx, ly, lz, n.shape, n.params);
                break;
            case pt::renderer::SDF_OP_SMOOTH_UNION:
                d = OpSmoothUnion(dists[n.child_a], dists[n.child_b], n.params[0]);
                break;
            case pt::renderer::SDF_OP_SMOOTH_SUBTRACT:
                d = OpSmoothSubtract(dists[n.child_a], dists[n.child_b], n.params[0]);
                break;
            case pt::renderer::SDF_OP_SMOOTH_INTERSECT:
                d = OpSmoothIntersect(dists[n.child_a], dists[n.child_b], n.params[0]);
                break;
            case pt::renderer::SDF_OP_DISPLACE:
                d = OpDisplace(dists[n.child_a], n.params[0]);
                break;
            default:
                d = 1e30f;
                break;
        }
        dists[i] = d;
    }
    return dists[c.node_count - 1u];
}

// -----------------------------------------------------------------------------
// Möller-Trumbore ray-triangle intersection, +X-ray flavour. Returns
// true if the ray (origin, +X direction) crosses the triangle in front
// of the origin. Used by the mesh inside-test below.
// -----------------------------------------------------------------------------
bool RayTriPosX(float ox, float oy, float oz,
                const float v0[3], const float v1[3], const float v2[3]) {
    // edge vectors
    const float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
    const float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
    // h = cross(d, e2). d = (1, 0, 0). So h = (0 * e2z - 0 * e2y,
    //                                          0 * e2x - 1 * e2z,
    //                                          1 * e2y - 0 * e2x)
    //                                       = (0, -e2z, e2y)
    const float hx = 0.0f;
    const float hy = -e2z;
    const float hz =  e2y;
    const float a  = e1x*hx + e1y*hy + e1z*hz;  // dot(e1, h)
    if (std::fabs(a) < 1e-12f) return false;    // ray parallel to triangle
    const float inv_a = 1.0f / a;
    // s = origin - v0
    const float sx = ox - v0[0];
    const float sy = oy - v0[1];
    const float sz = oz - v0[2];
    const float u  = inv_a * (sx*hx + sy*hy + sz*hz);
    if (u < 0.0f || u > 1.0f) return false;
    // q = cross(s, e1) = (sy*e1z - sz*e1y, sz*e1x - sx*e1z, sx*e1y - sy*e1x)
    const float qx = sy*e1z - sz*e1y;
    const float qy = sz*e1x - sx*e1z;
    const float qz = sx*e1y - sy*e1x;
    // v = inv_a * dot(d, q). dot((1,0,0), q) = qx.
    const float v  = inv_a * qx;
    if (v < 0.0f || u + v > 1.0f) return false;
    // t = inv_a * dot(e2, q)
    const float t  = inv_a * (e2x*qx + e2y*qy + e2z*qz);
    return t > 1e-6f;  // strictly in front of origin (skip self-hits)
}

bool PointInMesh(float px, float py, float pz,
                 const float* positions,
                 const std::uint32_t* indices,
                 std::size_t index_count) {
    // Even-odd parity count of +X-ray triangle crossings. Robust for
    // closed manifold meshes; degenerate cases (ray grazing an edge)
    // cancel out in pairs across the bitmap.
    std::uint32_t hits = 0u;
    for (std::size_t i = 0; i + 2 < index_count; i += 3) {
        const float* a = &positions[std::size_t(indices[i + 0]) * 3];
        const float* b = &positions[std::size_t(indices[i + 1]) * 3];
        const float* c = &positions[std::size_t(indices[i + 2]) * 3];
        if (RayTriPosX(px, py, pz, a, b, c)) ++hits;
    }
    return (hits & 1u) != 0u;
}

inline double NowSeconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// -----------------------------------------------------------------------------
// VoxelizeSphere
// -----------------------------------------------------------------------------
bool VoxelizeSphere(std::uint32_t object_id,
                    const float center[3], float radius,
                    float voxel_size,
                    const VoxelMaterial& material,
                    VoxelGrid& out) {
    if (radius <= 0.0f) {
        LOG_ERROR("[voxel] sphere id={} radius {} must be > 0", object_id, radius);
        return false;
    }
    if (voxel_size <= 0.0f) {
        LOG_ERROR("[voxel] sphere id={} voxel_size {} must be > 0", object_id, voxel_size);
        return false;
    }
    const float aabb_min[3] = { center[0] - radius, center[1] - radius, center[2] - radius };
    const float aabb_max[3] = { center[0] + radius, center[1] + radius, center[2] + radius };
    out.Init(object_id, aabb_min, aabb_max, voxel_size, material);

    const double t0 = NowSeconds();
    const float half = 0.5f * voxel_size;
    for (std::uint32_t k = 0; k < out.DimZ(); ++k)
    for (std::uint32_t j = 0; j < out.DimY(); ++j)
    for (std::uint32_t i = 0; i < out.DimX(); ++i) {
        auto c = out.VoxelCenter(i, j, k);
        const float d = SdSphere(c[0] - center[0],
                                 c[1] - center[1],
                                 c[2] - center[2], radius);
        if (d < half) out.Set(i, j, k, true);
    }
    out.RefreshOccupiedCount();
    const double dt = NowSeconds() - t0;
    LOG_INFO("[voxel] sphere id={} r={:.3f}m voxel={:.3f}m dims={}x{}x{} occupied={} in {:.3f}s",
             object_id, radius, voxel_size, out.DimX(), out.DimY(), out.DimZ(),
             out.Occupied(), dt);
    return out.Occupied() > 0u;
}

// -----------------------------------------------------------------------------
// VoxelizeSdf
// -----------------------------------------------------------------------------
bool VoxelizeSdf(std::uint32_t object_id,
                 const pt::renderer::SdfPrim& cluster,
                 float voxel_size,
                 const VoxelMaterial& material,
                 VoxelGrid& out) {
    if (voxel_size <= 0.0f) {
        LOG_ERROR("[voxel] sdf id={} voxel_size {} must be > 0", object_id, voxel_size);
        return false;
    }
    if (cluster.node_count == 0u) {
        LOG_ERROR("[voxel] sdf id={} has no nodes", object_id);
        return false;
    }
    out.Init(object_id, cluster.aabb_min, cluster.aabb_max, voxel_size, material);

    const double t0 = NowSeconds();
    const float half = 0.5f * voxel_size;
    for (std::uint32_t k = 0; k < out.DimZ(); ++k)
    for (std::uint32_t j = 0; j < out.DimY(); ++j)
    for (std::uint32_t i = 0; i < out.DimX(); ++i) {
        auto c = out.VoxelCenter(i, j, k);
        const float d = SdfClusterDist(cluster, c[0], c[1], c[2]);
        if (d < half) out.Set(i, j, k, true);
    }
    out.RefreshOccupiedCount();
    const double dt = NowSeconds() - t0;
    LOG_INFO("[voxel] sdf id={} voxel={:.3f}m dims={}x{}x{} occupied={} in {:.3f}s",
             object_id, voxel_size, out.DimX(), out.DimY(), out.DimZ(),
             out.Occupied(), dt);
    return out.Occupied() > 0u;
}

// -----------------------------------------------------------------------------
// VoxelizeMesh
// -----------------------------------------------------------------------------
bool VoxelizeMesh(std::uint32_t object_id,
                  const float* positions,
                  std::size_t  vertex_count,
                  const std::uint32_t* indices,
                  std::size_t  index_count,
                  float voxel_size,
                  const VoxelMaterial& material,
                  VoxelGrid& out) {
    if (positions == nullptr || vertex_count == 0u || index_count < 3u) {
        LOG_ERROR("[voxel] mesh id={} empty input (verts={} indices={})",
                  object_id, vertex_count, index_count);
        return false;
    }
    if (voxel_size <= 0.0f) {
        LOG_ERROR("[voxel] mesh id={} voxel_size {} must be > 0", object_id, voxel_size);
        return false;
    }
    // Bounding box of the mesh. Use FLT_MAX as the +/-infinity
    // sentinel: the project compiles with -ffast-math which disables
    // proper infinity support (-Wnan-infinity-disabled). Any vertex
    // value of magnitude < FLT_MAX overrides the sentinel cleanly.
    float aabb_min[3] = {  std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max() };
    float aabb_max[3] = { -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max() };
    for (std::size_t v = 0; v < vertex_count; ++v) {
        const float* p = &positions[v * 3];
        for (int a = 0; a < 3; ++a) {
            if (p[a] < aabb_min[a]) aabb_min[a] = p[a];
            if (p[a] > aabb_max[a]) aabb_max[a] = p[a];
        }
    }
    out.Init(object_id, aabb_min, aabb_max, voxel_size, material);

    const double t0 = NowSeconds();
    for (std::uint32_t k = 0; k < out.DimZ(); ++k)
    for (std::uint32_t j = 0; j < out.DimY(); ++j)
    for (std::uint32_t i = 0; i < out.DimX(); ++i) {
        auto c = out.VoxelCenter(i, j, k);
        if (PointInMesh(c[0], c[1], c[2], positions, indices, index_count)) {
            out.Set(i, j, k, true);
        }
    }
    out.RefreshOccupiedCount();
    const double dt = NowSeconds() - t0;
    LOG_INFO("[voxel] mesh id={} tris={} voxel={:.3f}m dims={}x{}x{} occupied={} in {:.3f}s",
             object_id, index_count / 3u, voxel_size,
             out.DimX(), out.DimY(), out.DimZ(), out.Occupied(), dt);
    return out.Occupied() > 0u;
}

bool VoxelizeBakedMesh(std::uint32_t object_id,
                       const pt::csg::BakedMesh& mesh,
                       float voxel_size,
                       const VoxelMaterial& material,
                       VoxelGrid& out) {
    if (mesh.Empty()) {
        LOG_ERROR("[voxel] mesh id={} BakedMesh is empty", object_id);
        return false;
    }
    return VoxelizeMesh(object_id,
                        mesh.positions.data(), mesh.VertexCount(),
                        mesh.indices.data(),   mesh.indices.size(),
                        voxel_size, material, out);
}

// -----------------------------------------------------------------------------
// MakeCachePath
// -----------------------------------------------------------------------------
std::string MakeCachePath(const std::string& dir_prefix,
                          std::uint32_t object_id,
                          float voxel_size,
                          std::uint64_t content_key) {
    const std::uint32_t vsize_micro =
        static_cast<std::uint32_t>(std::lround(double(voxel_size) * 1e6));
    char buf[256];
    if (content_key == 0ull) {
        std::snprintf(buf, sizeof(buf), "%s/voxel_cache_%u_%u.bin",
                      dir_prefix.c_str(), object_id, vsize_micro);
    } else {
        std::snprintf(buf, sizeof(buf), "%s/voxel_cache_%u_%u_%016llx.bin",
                      dir_prefix.c_str(), object_id, vsize_micro,
                      static_cast<unsigned long long>(content_key));
    }
    return std::string(buf);
}

}  // namespace pt::destruction

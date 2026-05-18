// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "AnalyticBvh.h"

#include "../core/Tracy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace pt::renderer {

// --- SDF Phase 1 (#97) -----------------------------------------------------
namespace {

// Per-leaf AABB in shape-local space (i.e. relative to node.center).
// Returns false if the leaf is degenerate (zero radius / extent).
bool LeafLocalAabb(const SdfNode& n, float (&mn)[3], float (&mx)[3]) {
    switch (n.shape) {
        case SDF_SHAPE_SPHERE: {
            float r = n.params[0];
            if (r <= 0.0f) return false;
            mn[0] = -r; mn[1] = -r; mn[2] = -r;
            mx[0] =  r; mx[1] =  r; mx[2] =  r;
            return true;
        }
        case SDF_SHAPE_BOX: {
            float hx = n.params[0], hy = n.params[1], hz = n.params[2];
            if (hx <= 0.0f || hy <= 0.0f || hz <= 0.0f) return false;
            mn[0] = -hx; mn[1] = -hy; mn[2] = -hz;
            mx[0] =  hx; mx[1] =  hy; mx[2] =  hz;
            return true;
        }
        case SDF_SHAPE_ROUNDED_BOX: {
            // Rounded box's surface lies outside the half-extents by the
            // corner radius, so widen the AABB by r in every direction.
            float hx = n.params[0] + n.params[3];
            float hy = n.params[1] + n.params[3];
            float hz = n.params[2] + n.params[3];
            if (hx <= 0.0f || hy <= 0.0f || hz <= 0.0f) return false;
            mn[0] = -hx; mn[1] = -hy; mn[2] = -hz;
            mx[0] =  hx; mx[1] =  hy; mx[2] =  hz;
            return true;
        }
        case SDF_SHAPE_TORUS: {
            // Torus laid in XZ plane (Y is the symmetry axis), major
            // radius R, minor radius r. Outer XZ radius is R+r; Y
            // extent is +/- r.
            float R = n.params[0];
            float r = n.params[1];
            if (R <= 0.0f || r <= 0.0f) return false;
            mn[0] = -(R + r); mn[1] = -r; mn[2] = -(R + r);
            mx[0] =  (R + r); mx[1] =  r; mx[2] =  (R + r);
            return true;
        }
        case SDF_SHAPE_CAPSULE: {
            // Capsule from (0, -h, 0) to (0, +h, 0) with radius r.
            float h = n.params[0];
            float r = n.params[1];
            if (h < 0.0f || r <= 0.0f) return false;
            mn[0] = -r; mn[1] = -h - r; mn[2] = -r;
            mx[0] =  r; mx[1] =  h + r; mx[2] =  r;
            return true;
        }
        case SDF_SHAPE_PLANE: {
            // Plane is infinite -- the host caller should not put a
            // plane SDF in a cluster (planes belong on the analytic-
            // primitive linear path). Refuse so we don't try to box-
            // bound infinity.
            return false;
        }
        // --- SDF Phase 3 (#99) fractal DEs --------------------------------
        //
        // All three fractals carry their "effective bound radius" in
        // params[1] -- the radius of the world-space sphere that
        // conservatively contains the fractal's iso-surface. We box-
        // bound that sphere (axis-aligned cube of side 2*R). The
        // exact iso-surface lies well inside this bound (the
        // Mandelbulb's reachable extent is ~1.2 m; the Mandelbox's
        // limit set sits within ~4 m of origin at the canonical
        // scale=2.5; Apollonian limit set at scale=1.3 stays within
        // ~2 m). Empty corners pay one sphere-trace AABB-exit hit
        // before the trace gives up.
        //
        // params[1] <= 0 is a host author error -- a fractal with no
        // bound radius would force a "default world" cube that turns
        // the sphere-trace into a full-screen tax. Refuse so the
        // failure surfaces as a clear parse error rather than a 1 fps
        // mystery.
        case SDF_SHAPE_MANDELBULB:
        case SDF_SHAPE_MANDELBOX:
        case SDF_SHAPE_APOLLONIAN: {
            float R = n.params[1];
            if (R <= 0.0f) return false;
            mn[0] = -R; mn[1] = -R; mn[2] = -R;
            mx[0] =  R; mx[1] =  R; mx[2] =  R;
            return true;
        }
        // --- end SDF Phase 3 ---
    }
    return false;
}

// Recursive AABB walker. Returns false if a malformed reference is hit.
bool NodeAabb(const SdfPrim& prim, std::uint32_t idx,
              float (&mn)[3], float (&mx)[3]) {
    if (idx >= prim.node_count) return false;
    const SdfNode& n = prim.nodes[idx];
    auto translate = [&](float (&a)[3], float (&b)[3]) {
        for (int i = 0; i < 3; ++i) { a[i] += n.center[i]; b[i] += n.center[i]; }
    };
    auto widen = [](float (&a)[3], float (&b)[3], float k) {
        for (int i = 0; i < 3; ++i) { a[i] -= k; b[i] += k; }
    };
    auto union_aabb = [](float (&a)[3], float (&b)[3],
                         const float (&u)[3], const float (&v)[3]) {
        for (int i = 0; i < 3; ++i) {
            a[i] = std::min(a[i], u[i]);
            b[i] = std::max(b[i], v[i]);
        }
    };
    switch (n.op) {
        case SDF_OP_LEAF: {
            float lmn[3], lmx[3];
            if (!LeafLocalAabb(n, lmn, lmx)) return false;
            for (int i = 0; i < 3; ++i) { mn[i] = lmn[i]; mx[i] = lmx[i]; }
            translate(mn, mx);
            return true;
        }
        case SDF_OP_SMOOTH_UNION:
        case SDF_OP_SMOOTH_INTERSECT: {
            // Smooth blend can bulge the surface outward by ~k. Union
            // bounds = union of children's bounds + k pad. Intersect
            // bounds = the children's overlap widened by k -- but the
            // raw `max-min / min-max` formula yields an INVALID AABB
            // (mn > mx on some axis) when the children's bounds are
            // disjoint on that axis. The downstream slab test treats
            // an inverted box as a miss, which is technically correct
            // for a hard intersect of disjoint shapes (no surface) but
            // the smooth-intersect blend band CAN poke surface up to k
            // outside each child's bound. Fall back to the children's
            // UNION (widened by k below) in that disjoint case -- it's
            // a conservative superset that's guaranteed to contain any
            // smooth-intersect surface, and only one or two extra
            // sphere-trace steps wasted on empty space.
            float amn[3], amx[3], bmn[3], bmx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            if (!NodeAabb(prim, n.child_b, bmn, bmx)) return false;
            if (n.op == SDF_OP_SMOOTH_UNION) {
                for (int i = 0; i < 3; ++i) {
                    mn[i] = amn[i]; mx[i] = amx[i];
                }
                union_aabb(mn, mx, bmn, bmx);
            } else { // SMOOTH_INTERSECT
                bool disjoint = false;
                for (int i = 0; i < 3; ++i) {
                    mn[i] = std::max(amn[i], bmn[i]);
                    mx[i] = std::min(amx[i], bmx[i]);
                    if (mn[i] > mx[i]) disjoint = true;
                }
                if (disjoint) {
                    // No overlap on at least one axis: hard-intersect
                    // surface is empty. Use the children's union as a
                    // safe superset of any smooth-blend surface that
                    // might bulge between them.
                    for (int i = 0; i < 3; ++i) {
                        mn[i] = amn[i]; mx[i] = amx[i];
                    }
                    union_aabb(mn, mx, bmn, bmx);
                }
            }
            widen(mn, mx, std::max(0.0f, n.params[0]));
            return true;
        }
        case SDF_OP_SMOOTH_SUBTRACT: {
            // a - b: result is at most a, possibly carved smaller.
            // Reuse a's AABB widened by k (the smooth carve can bulge
            // by k along the cut boundary).
            float amn[3], amx[3], bmn[3], bmx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            if (!NodeAabb(prim, n.child_b, bmn, bmx)) return false;
            for (int i = 0; i < 3; ++i) {
                mn[i] = amn[i]; mx[i] = amx[i];
            }
            widen(mn, mx, std::max(0.0f, n.params[0]));
            return true;
        }
        case SDF_OP_DISPLACE: {
            float amn[3], amx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            for (int i = 0; i < 3; ++i) { mn[i] = amn[i]; mx[i] = amx[i]; }
            widen(mn, mx, std::abs(n.params[0]));
            return true;
        }
        // --- SDF Phase 2 (#98) procedural / noise / domain ops -------------
        case SDF_OP_DISPLACE_NOISE: {
            // Noise band is centred in [-amp, +amp]. Widen the child's
            // AABB by amp so the displaced surface stays inside the
            // bound -- the trace's slab test is the only thing
            // preventing wasted iterations on empty space.
            float amn[3], amx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            for (int i = 0; i < 3; ++i) { mn[i] = amn[i]; mx[i] = amx[i]; }
            widen(mn, mx, std::abs(n.params[0]));
            return true;
        }
        case SDF_OP_TWIST:
        case SDF_OP_BEND: {
            // Twist / bend rotate the child's local frame. The exact
            // worst-case extent is hard to express without a per-axis
            // arc tracing; we conservatively widen to a ball that
            // contains all rotations of the child's AABB about its
            // own centre (rotation never moves points farther than
            // the AABB's half-diagonal from the original surface).
            // This is a true superset by construction.
            float amn[3], amx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            float cx = 0.5f * (amn[0] + amx[0]);
            float cy = 0.5f * (amn[1] + amx[1]);
            float cz = 0.5f * (amn[2] + amx[2]);
            float ex = 0.5f * (amx[0] - amn[0]);
            float ey = 0.5f * (amx[1] - amn[1]);
            float ez = 0.5f * (amx[2] - amn[2]);
            float r = std::sqrt(ex * ex + ey * ey + ez * ez);
            mn[0] = cx - r; mx[0] = cx + r;
            mn[1] = cy - r; mx[1] = cy + r;
            mn[2] = cz - r; mx[2] = cz + r;
            return true;
        }
        case SDF_OP_REPEAT: {
            // Infinite domain repetition. The surface tiles space
            // forever, so no finite AABB is conservatively correct
            // for the full repeat. The host wraps this with a
            // SDF_OP_REPEAT_LIMITED (or carves it via a parent
            // smooth-intersect with a finite box) in practice. As a
            // fallback for an explicit infinite repeat, we cap the
            // bound at +/-1000 m on each axis where the period is
            // non-zero -- well past the typical scene extent. The
            // engine logs a warning when this fallback fires.
            float amn[3], amx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            for (int i = 0; i < 3; ++i) {
                if (n.params[i] > 1e-6f) {
                    mn[i] = -1000.0f;
                    mx[i] =  1000.0f;
                } else {
                    mn[i] = amn[i];
                    mx[i] = amx[i];
                }
            }
            return true;
        }
        case SDF_OP_REPEAT_LIMITED: {
            // Bounded repetition. Cell count = 2 * limit + 1 on each
            // axis (cells run from -limit to +limit inclusive). Each
            // cell is the child's bound translated by k * period.
            float amn[3], amx[3];
            if (!NodeAabb(prim, n.child_a, amn, amx)) return false;
            // Decode the packed int3 limit from params[3]. Same
            // 10-bit lane layout as the shader.
            std::uint32_t packed{};
            std::memcpy(&packed, &n.params[3], sizeof(float));
            int lim[3] = {
                int( packed         & 0x3ffu),
                int((packed >> 10) & 0x3ffu),
                int((packed >> 20) & 0x3ffu),
            };
            for (int i = 0; i < 3; ++i) {
                if (n.params[i] > 1e-6f) {
                    float span = float(lim[i]) * n.params[i];
                    mn[i] = amn[i] - span;
                    mx[i] = amx[i] + span;
                } else {
                    mn[i] = amn[i];
                    mx[i] = amx[i];
                }
            }
            return true;
        }
        // --- end SDF Phase 2 ----------------------------------------------
    }
    return false;
}

}  // namespace

bool ComputeSdfAabb(SdfPrim& prim) {
    if (prim.node_count == 0 || prim.node_count > SdfPrim::kMaxNodes) return false;
    float mn[3], mx[3];
    // Convention: the LAST node in the flat array is the root of the
    // merged op-tree (matches the shader's `dists[node_count - 1]` read
    // in sdfClusterDist / sdfClusterNormal). Walk from there so the
    // AABB reflects the actual expression, not just child A.
    if (!NodeAabb(prim, prim.node_count - 1u, mn, mx)) return false;
    for (int i = 0; i < 3; ++i) {
        prim.aabb_min[i] = mn[i];
        prim.aabb_max[i] = mx[i];
    }
    return true;
}
// --- end SDF Phase 1 -------------------------------------------------------

namespace {

inline void AabbInclude(float (&mn)[3], float (&mx)[3],
                        const float (&p)[3], float r) {
    for (int i = 0; i < 3; ++i) {
        if (p[i] - r < mn[i]) mn[i] = p[i] - r;
        if (p[i] + r > mx[i]) mx[i] = p[i] + r;
    }
}

inline void AabbReset(float (&mn)[3], float (&mx)[3]) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    mn[0] = mn[1] = mn[2] =  inf;
    mx[0] = mx[1] = mx[2] = -inf;
}

}  // namespace

void AnalyticBvh::Build(std::span<const BvhPrim> prims) {
    PT_ZONE_SCOPED_N("AnalyticBvh::Build");
    nodes_.clear();
    permuted_.clear();
    working_.clear();
    if (prims.empty()) return;

    working_.reserve(prims.size());
    for (const auto& p : prims) {
        WorkPrim wp{};
        wp.prim       = p;
        wp.centroid[0] = p.center[0];
        wp.centroid[1] = p.center[1];
        wp.centroid[2] = p.center[2];
        working_.push_back(wp);
    }

    // Upper bound on internal+leaf nodes is 2N-1 for N leaf primitives
    // when leaf size is 1. With kMaxLeafSize the count is smaller, but
    // we reserve generously and shrink later.
    nodes_.reserve(2 * working_.size());
    nodes_.push_back({});       // root slot
    BuildRecursive(0, 0, static_cast<std::uint32_t>(working_.size()));

    // Emit the permuted prim-id sequence from the (now reordered)
    // working_ list. The caller permutes its GPU primitive buffer to
    // match so leaf.left_first directly indexes the buffer.
    permuted_.reserve(working_.size());
    for (const auto& wp : working_) permuted_.push_back(wp.prim.prim_id);
}

std::uint32_t AnalyticBvh::BuildRecursive(std::uint32_t node_idx,
                                          std::uint32_t first,
                                          std::uint32_t count) {
    // IMPORTANT: never bind a reference into nodes_ across a
    // nodes_.push_back() call. vector reallocation can move the
    // backing storage, leaving the reference dangling -- internal
    // node writes after the child push_backs were corrupting memory
    // (rendered as empty BVH on first attempt). Compute the AABB
    // into locals and copy to nodes_[node_idx] each time we touch it.
    float bn[3], bx[3];
    AabbReset(bn, bx);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& wp = working_[first + i];
        AabbInclude(bn, bx, wp.prim.center, wp.prim.radius);
    }
    for (int a = 0; a < 3; ++a) {
        nodes_[node_idx].aabb_min[a] = bn[a];
        nodes_[node_idx].aabb_max[a] = bx[a];
    }

    if (count <= kMaxLeafSize) {
        nodes_[node_idx].left_first = first;
        nodes_[node_idx].count      = count;
        return 1;
    }

    // Choose split axis = axis with widest spread of centroids.
    float centroid_mn[3];
    float centroid_mx[3];
    AabbReset(centroid_mn, centroid_mx);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& c = working_[first + i].centroid;
        for (int a = 0; a < 3; ++a) {
            if (c[a] < centroid_mn[a]) centroid_mn[a] = c[a];
            if (c[a] > centroid_mx[a]) centroid_mx[a] = c[a];
        }
    }
    int   axis  = 0;
    float widest = centroid_mx[0] - centroid_mn[0];
    for (int a = 1; a < 3; ++a) {
        float w = centroid_mx[a] - centroid_mn[a];
        if (w > widest) { widest = w; axis = a; }
    }

    // Degenerate: all centroids coincide (e.g., concentric spheres at
    // the same origin). Centroid-based split would recurse forever.
    // Fall back to a simple index-median split so leaf sizes still
    // stay bounded by ~kMaxLeafSize log-depth-down. The shader-side
    // per-leaf loop relies on this invariant -- a leaf with hundreds
    // of prims would defeat the whole point of the BVH.
    if (widest <= 0.0f) {
        // No partitioning needed; the range is already arbitrary since
        // every prim has the same centroid. Just slice it down the
        // middle by index.
    } else {
        // Median split: partition around the median centroid on `axis`.
        // nth_element is O(N) and good enough; SAH would build a slightly
        // tighter tree but the build cost on edits would rise nontrivially
        // for a marginal trace-time win on the prim counts we expect.
        auto* begin = working_.data() + first;
        auto* mid   = begin + count / 2;
        auto* end   = begin + count;
        std::nth_element(begin, mid, end,
            [axis](const WorkPrim& a, const WorkPrim& b) {
                return a.centroid[axis] < b.centroid[axis];
            });
    }

    const std::uint32_t left_count  = count / 2;
    const std::uint32_t right_count = count - left_count;

    // Reserve two child slots; left child first, right child immediately
    // after. left_first on the internal node records the left child's
    // index. Write to the internal node BEFORE the push_backs so any
    // reallocation after this point is harmless to it.
    const std::uint32_t left_idx  = static_cast<std::uint32_t>(nodes_.size());
    nodes_[node_idx].left_first = left_idx;
    nodes_[node_idx].count      = 0;   // 0 marks internal
    nodes_.push_back({});
    const std::uint32_t right_idx = left_idx + 1u;
    nodes_.push_back({});

    std::uint32_t emitted = 1;
    emitted += BuildRecursive(left_idx,  first,              left_count);
    emitted += BuildRecursive(right_idx, first + left_count, right_count);
    return emitted;
}

}  // namespace pt::renderer

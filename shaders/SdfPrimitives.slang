// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SDF Phase 1 (#97) -- analytic signed-distance-field primitive
// library. Used by PathTrace.slang when a BVH leaf is tagged as
// SHAPE_SDF. The host (src/renderer/AnalyticBvh.{h,cpp}) builds the
// flat per-cluster node table; this module walks the table, runs the
// sphere-trace inside the cluster AABB, and returns hit position +
// analytic-gradient normal + material id.
//
// Conventions:
//   * All distances and parameters are in METRES (world units). No
//     demo-scaled tricks -- a 1 m sphere literally has radius 1.0.
//   * Per-shape normals are computed from the closed-form gradient.
//     Phase 1 deliberately avoids central differences (those are
//     Phase 2's deal once procedural noise / displacement enters).
//   * The op-tree evaluates with a tiny register stack of distances;
//     SdfPrim::kMaxNodes is 8 on the host so the shader can keep the
//     stack in registers without spilling.
//
// References:
//   * Inigo Quilez, "Distance functions" -- the standard closed-form
//     library (sdSphere / sdBox / sdRoundBox / sdTorus / sdCapsule /
//     sdPlane formulas below match his canon).
//   * Hart, "Sphere tracing: A geometric method for the antialiased
//     ray tracing of implicit surfaces" (1996) -- the trace loop.

// Wire-format constants. MUST match SdfShape / SdfOp in
// src/renderer/AnalyticBvh.h. Renumbering one without the other
// silently corrupts cluster evaluation.
static const uint SDF_SHAPE_SPHERE      = 0u;
static const uint SDF_SHAPE_BOX         = 1u;
static const uint SDF_SHAPE_ROUNDED_BOX = 2u;
static const uint SDF_SHAPE_TORUS       = 3u;
static const uint SDF_SHAPE_CAPSULE     = 4u;
static const uint SDF_SHAPE_PLANE       = 5u;

static const uint SDF_OP_LEAF             = 0u;
static const uint SDF_OP_SMOOTH_UNION     = 1u;
static const uint SDF_OP_SMOOTH_SUBTRACT  = 2u;
static const uint SDF_OP_SMOOTH_INTERSECT = 3u;
static const uint SDF_OP_DISPLACE         = 4u;

static const uint SDF_MAX_NODES = 8u;     // mirrors SdfPrim::kMaxNodes

// ----- per-shape signed distance functions ---------------------------------
//
// Each function evaluates the SDF for a point p expressed in shape-LOCAL
// space (the host applies the per-node center translation before
// calling). All formulas are exact distance fields, NOT approximations
// -- the sphere-trace's per-step `t += d` advance stays correct.

float sdSphere(float3 p, float r) {
    return length(p) - r;
}

float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, float3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundBox(float3 p, float3 b, float r) {
    float3 q = abs(p) - b;
    return length(max(q, float3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Torus laid in the XZ plane (Y is the symmetry axis). R = major
// radius from origin to ring center; r = minor radius (tube radius).
float sdTorus(float3 p, float R, float r) {
    float2 q = float2(length(float2(p.x, p.z)) - R, p.y);
    return length(q) - r;
}

// Capsule along the +Y axis from (0,-h,0) to (0,+h,0) with radius r.
float sdCapsule(float3 p, float h, float r) {
    float y = clamp(p.y, -h, h);
    float3 q = float3(p.x, p.y - y, p.z);
    return length(q) - r;
}

// Plane with unit normal n through origin offset d:  n . p + d = 0.
float sdPlane(float3 p, float3 n, float d) {
    return dot(p, n) + d;
}

// ----- per-shape analytic NORMALS (closed-form gradient) -------------------
//
// One per shape so we never have to fall back to central differences in
// Phase 1. Each returns a UNIT vector pointing outward from the surface
// at point p (local frame). Caller transforms back to world by passing
// the same local-frame p that was used to evaluate the distance.

float3 sdSphereNormal(float3 p) {
    float l = length(p);
    return l > 1e-12 ? p / l : float3(0, 1, 0);
}

float3 sdBoxNormal(float3 p, float3 b) {
    // Gradient of sdBox at p: outward face normal at the nearest point
    // on the box surface. For points OUTSIDE the box, the gradient is
    // along max(|p|-b, 0); for points INSIDE (rare during a forward
    // sphere-trace) it's the axis of the smallest |p_i| - b_i (negative).
    float3 q  = abs(p) - b;
    float3 sg = sign(p);
    bool   outside = max(q.x, max(q.y, q.z)) > 0.0;
    if (outside) {
        float3 m = max(q, float3(0.0));
        float lm = length(m);
        if (lm < 1e-12) return sg * float3(0, 1, 0);
        return (m / lm) * sg;
    }
    // Inside: pick the axis of the smallest q (closest face).
    float3 n = float3(0.0);
    if (q.x > q.y && q.x > q.z) n.x = sg.x;
    else if (q.y > q.z)         n.y = sg.y;
    else                         n.z = sg.z;
    return n;
}

float3 sdRoundBoxNormal(float3 p, float3 b) {
    // Round-box gradient is the same as box gradient at the inner
    // half-extents (the rounded corner just shifts the surface outward
    // by r along that gradient direction).
    return sdBoxNormal(p, b);
}

float3 sdTorusNormal(float3 p, float R) {
    // Implicit (length(xz) - R)^2 + y^2 = r^2. Gradient direction is
    // (p - closest-point-on-ring). The closest point on the ring is at
    // distance R from origin along the projection of p into the XZ
    // plane.
    float2 pxz = float2(p.x, p.z);
    float  lxz = length(pxz);
    if (lxz < 1e-12) return float3(0, sign(p.y), 0);
    float2 ringxz = pxz * (R / lxz);
    float3 ring   = float3(ringxz.x, 0.0, ringxz.y);
    float3 d      = p - ring;
    float  ld     = length(d);
    return ld > 1e-12 ? d / ld : float3(0, 1, 0);
}

float3 sdCapsuleNormal(float3 p, float h) {
    float  y = clamp(p.y, -h, h);
    float3 q = p - float3(0.0, y, 0.0);
    float  l = length(q);
    return l > 1e-12 ? q / l : float3(0, 1, 0);
}

float3 sdPlaneNormal(float3 n) {
    return n;
}

// ----- smooth-CSG ops ------------------------------------------------------
//
// Each returns a SIGNED distance that's exact within the linearity of
// the underlying primitives -- inside the smoothing band of width k
// they're approximate by design (that's what "smooth" means). The
// classic IQ polynomial smin / smax / sdiff formulas; see
//   https://iquilezles.org/articles/smin/
//
// Note Phase 1 does not implement opDisplace as a noise term -- the
// host's `displace` op just adds a constant bump scaled by the
// `amplitude` parameter so the test surface for the op is visible.
// Phase 2 swaps the constant for real procedural noise.

float opSmoothUnion(float d1, float d2, float k) {
    if (k <= 0.0) return min(d1, d2);
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return lerp(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtract(float d1, float d2, float k) {
    // d1 carved by d2. Inigo's polynomial form (k > 0):
    if (k <= 0.0) return max(d1, -d2);
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return lerp(d1, -d2, h) + k * h * (1.0 - h);
}

float opSmoothIntersect(float d1, float d2, float k) {
    if (k <= 0.0) return max(d1, d2);
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return lerp(d2, d1, h) + k * h * (1.0 - h);
}

// Constant-bump displace. Phase 1 placeholder; Phase 2 will swap the
// constant for a procedural-noise term (3D simplex / FBM). This still
// gives the op a visible effect so test scenes can exercise the wire
// format.
float opDisplace(float d, float amp) {
    return d + amp;
}

// ----- per-node distance + (optional) normal evaluation --------------------
//
// `evalLeafDist` returns the distance for a LEAF node only. Op nodes
// combine their children's already-computed distances via the smooth-
// CSG ops above.
float evalLeafDist(float3 p_local, uint shape, float4 params) {
    if      (shape == SDF_SHAPE_SPHERE)
        return sdSphere(p_local, params.x);
    else if (shape == SDF_SHAPE_BOX)
        return sdBox(p_local, params.xyz);
    else if (shape == SDF_SHAPE_ROUNDED_BOX)
        return sdRoundBox(p_local, params.xyz, params.w);
    else if (shape == SDF_SHAPE_TORUS)
        return sdTorus(p_local, params.x, params.y);
    else if (shape == SDF_SHAPE_CAPSULE)
        return sdCapsule(p_local, params.x, params.y);
    else if (shape == SDF_SHAPE_PLANE)
        return sdPlane(p_local, params.xyz, params.w);
    return 1e30;
}

float3 evalLeafNormal(float3 p_local, uint shape, float4 params) {
    if      (shape == SDF_SHAPE_SPHERE)
        return sdSphereNormal(p_local);
    else if (shape == SDF_SHAPE_BOX)
        return sdBoxNormal(p_local, params.xyz);
    else if (shape == SDF_SHAPE_ROUNDED_BOX)
        return sdRoundBoxNormal(p_local, params.xyz);
    else if (shape == SDF_SHAPE_TORUS)
        return sdTorusNormal(p_local, params.x);
    else if (shape == SDF_SHAPE_CAPSULE)
        return sdCapsuleNormal(p_local, params.x);
    else if (shape == SDF_SHAPE_PLANE)
        return sdPlaneNormal(params.xyz);
    return float3(0, 1, 0);
}

// ----- op-tree distance + analytic normal ---------------------------------
//
// SdfCluster is the GPU representation of one host SdfPrim. The op-
// tree is in `nodes[]`; root is at index 0. Material+albedo+AABB are
// in the header. Layout MUST match the SoA packing the host writes into
// the SDF storage buffer (see EnsureSdfPrimsUploaded in Engine.cpp).

struct SdfClusterNode {
    uint   op;
    uint   shape;
    uint   child_a;
    uint   child_b;
    float4 params;
    float3 center;
    uint   _pad;
};

struct SdfCluster {
    SdfClusterNode nodes[SDF_MAX_NODES];
    uint   node_count;
    uint   material;
    float3 aabb_min;
    float3 aabb_max;
    float3 albedo;
    float  roughness;
    float  ior;
};

// Evaluate distance from world-space point p to the surface encoded by
// the cluster. Iterative post-order pseudo-walk: nodes have been
// emitted by the host such that for any op-node, its children come at
// smaller indices, so a simple ascending walk fills a small per-node
// distance buffer that downstream ops index into.
float sdfClusterDist(SdfCluster c, float3 p) {
    float dists[SDF_MAX_NODES];
    [unroll(SDF_MAX_NODES)]
    for (uint i = 0u; i < SDF_MAX_NODES; ++i) {
        if (i >= c.node_count) break;
        SdfClusterNode n = c.nodes[i];
        float d;
        if (n.op == SDF_OP_LEAF) {
            d = evalLeafDist(p - n.center, n.shape, n.params);
        } else if (n.op == SDF_OP_SMOOTH_UNION) {
            d = opSmoothUnion(dists[n.child_a], dists[n.child_b], n.params.x);
        } else if (n.op == SDF_OP_SMOOTH_SUBTRACT) {
            d = opSmoothSubtract(dists[n.child_a], dists[n.child_b], n.params.x);
        } else if (n.op == SDF_OP_SMOOTH_INTERSECT) {
            d = opSmoothIntersect(dists[n.child_a], dists[n.child_b], n.params.x);
        } else if (n.op == SDF_OP_DISPLACE) {
            d = opDisplace(dists[n.child_a], n.params.x);
        } else {
            d = 1e30;
        }
        dists[i] = d;
    }
    return c.node_count > 0u ? dists[c.node_count - 1u] : 1e30;
}

// Analytic gradient of the cluster distance at p. For LEAF nodes we
// have closed forms (above). For smooth-CSG ops the gradient is the
// per-child gradient weighted by the polynomial blend factor h. This
// is the same `h` that appears in the IQ smin formula -- computing it
// once and reusing it costs one extra clamp per op.
float3 sdfClusterNormal(SdfCluster c, float3 p) {
    float  dists[SDF_MAX_NODES];
    float3 grads[SDF_MAX_NODES];
    [unroll(SDF_MAX_NODES)]
    for (uint i = 0u; i < SDF_MAX_NODES; ++i) {
        if (i >= c.node_count) break;
        SdfClusterNode n = c.nodes[i];
        float  d;
        float3 g;
        if (n.op == SDF_OP_LEAF) {
            float3 pl = p - n.center;
            d = evalLeafDist(pl, n.shape, n.params);
            g = evalLeafNormal(pl, n.shape, n.params);
        } else if (n.op == SDF_OP_SMOOTH_UNION) {
            float k  = max(n.params.x, 1e-6);
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            float h  = clamp(0.5 + 0.5 * (db - da) / k, 0.0, 1.0);
            d = lerp(db, da, h) - k * h * (1.0 - h);
            g = normalize(lerp(grads[n.child_b], grads[n.child_a], h));
        } else if (n.op == SDF_OP_SMOOTH_SUBTRACT) {
            float k  = max(n.params.x, 1e-6);
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            float h  = clamp(0.5 - 0.5 * (db + da) / k, 0.0, 1.0);
            d = lerp(da, -db, h) + k * h * (1.0 - h);
            g = normalize(lerp(grads[n.child_a], -grads[n.child_b], h));
        } else if (n.op == SDF_OP_SMOOTH_INTERSECT) {
            float k  = max(n.params.x, 1e-6);
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            float h  = clamp(0.5 - 0.5 * (db - da) / k, 0.0, 1.0);
            d = lerp(db, da, h) + k * h * (1.0 - h);
            g = normalize(lerp(grads[n.child_b], grads[n.child_a], h));
        } else if (n.op == SDF_OP_DISPLACE) {
            // Constant displace doesn't change the gradient direction.
            d = opDisplace(dists[n.child_a], n.params.x);
            g = grads[n.child_a];
        } else {
            d = 1e30; g = float3(0, 1, 0);
        }
        dists[i] = d;
        grads[i] = g;
    }
    return c.node_count > 0u ? grads[c.node_count - 1u] : float3(0, 1, 0);
}

// ----- ray-AABB entry / exit ----------------------------------------------
//
// Returns (t_enter, t_exit) for the slab intersection of ray
// (ro + rd * t) against the cluster AABB. If the ray misses the box
// entirely, t_exit < t_enter. Caller starts the sphere-trace at
// max(t_enter, 0) and stops at t_exit.
float2 sdfClusterAabbT(SdfCluster c, float3 ro, float3 rd_inv) {
    float3 t0 = (c.aabb_min - ro) * rd_inv;
    float3 t1 = (c.aabb_max - ro) * rd_inv;
    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);
    float  t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float  t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);
    return float2(t_enter, t_exit);
}

// ----- sphere-trace --------------------------------------------------------
//
// Single-cluster sphere-trace. Caller is responsible for first running
// the BVH walk and arriving at this cluster -- by then the ray is known
// to be inside (or about to enter) the cluster AABB. Returns true if a
// hit was found within [t_min, t_max]; out_t / out_normal carry the hit
// data on success.
//
// max_iters caps the inner loop and is sourced from the host cvar
// `r_sdf_max_iters` (default 128); epsilon comes from `r_sdf_epsilon`
// (default 1e-4 metres). These are passed via the host's PtPush struct.
bool sdfClusterTrace(SdfCluster c, float3 ro, float3 rd,
                     float t_min, float t_max,
                     uint max_iters, float epsilon,
                     out float out_t, out float3 out_normal) {
    out_t      = t_max;
    out_normal = float3(0, 1, 0);

    // Slab test to find the (entry, exit) interval inside the cluster
    // AABB. Skip the trace entirely if the ray misses the box.
    float3 rd_safe = select(abs(rd) > float3(1e-30), rd, float3(1e-30));
    float3 rd_inv  = 1.0 / rd_safe;
    float2 ts      = sdfClusterAabbT(c, ro, rd_inv);
    float  t       = max(ts.x, t_min);
    float  t_end   = min(ts.y, t_max);
    if (t >= t_end) return false;

    // Step bound: cap max_iters at SDF_MAX_TRACE_ITERS to keep an upper
    // bound on the shader's worst-case work. 256 is a generous Phase 1
    // ceiling for analytic primitives + smooth ops (fractals in Phase 3
    // will need more).
    uint iters = min(max_iters, 256u);
    for (uint i = 0u; i < iters; ++i) {
        float3 p = ro + rd * t;
        float  d = sdfClusterDist(c, p);
        if (d < epsilon) {
            out_t      = t;
            out_normal = normalize(sdfClusterNormal(c, p));
            return true;
        }
        t += d;
        if (t >= t_end) return false;
    }
    return false;
}

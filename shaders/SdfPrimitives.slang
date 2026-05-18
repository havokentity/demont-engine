// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// SDF Phase 1 (#97) -- analytic signed-distance-field primitive
// library. Phase 1 invokes this module from PathTrace.slang via a
// separate linear scan over all SDF clusters (one sphere-trace per
// cluster AABB). Folding SDF clusters into the analytic BVH as
// SHAPE_SDF leaves is deferred to a future phase (tracked in #111).
// The host (src/renderer/AnalyticBvh.{h,cpp}) builds the flat
// per-cluster node table; this module walks the table, runs the
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
//
// ----- compile-time gates --------------------------------------------------
// PT_SDF_PROCEDURAL_OPS  -- when defined to 1, Phase 2 (#98) procedural ops
//                           (sdf_displace_noise / twist / bend / repeat /
//                           repeat-limited) and their noise primitives +
//                           warp + scalar walkers are compiled in. Default
//                           OFF: a perf bisect on the default scene found
//                           ~6 ms/frame regression from this block alone,
//                           even when no scene cluster uses procedural ops
//                           -- Slang's megakernel dispatch inlines every
//                           branch into the sphere-trace inner loop.
// PT_SDF_AUTODIFF        -- when defined to 1, the Dual3 forward-mode
//                           autodiff machinery (struct + per-shape Dual3
//                           SDFs + AD cluster walker + forward-AD normal)
//                           is compiled in. Only meaningful when
//                           PT_SDF_PROCEDURAL_OPS is also on (the AD path
//                           is exclusively used to derive normals through
//                           procedural ops). Default OFF: when this is OFF
//                           and the user sets r_sdf_normal_mode=0, the
//                           host falls back to central differences with a
//                           [warn] line emitted at engine startup.
//
// Builds that need procedural SDFs should pass `-DPT_SDF_PROCEDURAL_OPS=1`
// to slangc (CMakePresets / pt_compile_slang EXTRA_DEFINES). Builds that
// also want the forward-AD normal path add `-DPT_SDF_AUTODIFF=1`. Phase 1
// analytic CSG (sphere / box / torus / smin etc.) is always available.
#ifndef PT_SDF_PROCEDURAL_OPS
#define PT_SDF_PROCEDURAL_OPS 0
#endif
#ifndef PT_SDF_AUTODIFF
#define PT_SDF_AUTODIFF 0
#endif
// PT_SDF_AUTODIFF without PT_SDF_PROCEDURAL_OPS makes no sense -- the AD
// walker only ever runs on procedural clusters. Promote the gate so we
// don't have to repeat the conjunction at every call site.
#if PT_SDF_AUTODIFF && !PT_SDF_PROCEDURAL_OPS
#undef PT_SDF_AUTODIFF
#define PT_SDF_AUTODIFF 0
#endif

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
// --- SDF Phase 2 (#98) ----------------------------------------------------
// Procedural ops. Numeric values 5+ are part of the host/shader wire
// format -- renumbering one without the other silently corrupts cluster
// evaluation (same caveat as Phase 1 ops above). These ops never have
// a *closed-form* analytic gradient -- the noise / repeat / twist / bend
// machinery composes through finite-difference or forward-AD numerics
// instead. See sdfClusterDistDN / sdfClusterDistDWarped below for the
// AD walker; sdfClusterDistProcN / sdfClusterDistWarped for the scalar
// distance walker the sphere tracer steps through.
//
//   SDF_OP_DISPLACE_NOISE  -- adds an FBM-noise band of amplitude
//                              params[0] (metres), base frequency
//                              params[1] (1/metres), N octaves
//                              packed as asuint(params[2]) (Phase 2
//                              clamps 1..6). Children: one child SDF
//                              whose surface is displaced.
//   SDF_OP_TWIST           -- twist child geometry around the Y axis
//                              with rate params[0] radians per metre
//                              of height. params[1..3] unused.
//   SDF_OP_BEND            -- bend child geometry along the X axis
//                              with rate params[0] radians per metre.
//                              params[1..3] unused.
//   SDF_OP_REPEAT          -- infinite domain repetition with period
//                              vec3 (params[0..2]). A zero period on
//                              an axis disables repetition on that
//                              axis.
//   SDF_OP_REPEAT_LIMITED  -- like SDF_OP_REPEAT but bounded by an
//                              integer half-extent vec3 packed as
//                              asuint(params[3]) into 10-bit lanes
//                              (xyz). Cell index is clamped to
//                              [-limit, limit] on each axis.
static const uint SDF_OP_DISPLACE_NOISE   = 5u;
static const uint SDF_OP_TWIST            = 6u;
static const uint SDF_OP_BEND             = 7u;
static const uint SDF_OP_REPEAT           = 8u;
static const uint SDF_OP_REPEAT_LIMITED   = 9u;
// --- end SDF Phase 2 ------------------------------------------------------

static const uint SDF_MAX_NODES = 8u;     // mirrors SdfPrim::kMaxNodes

// --- SDF Phase 2 (#98) ----------------------------------------------------
// Normal-evaluation mode constants. The host's r_sdf_normal_mode cvar
// selects between forward-mode autodiff and central differences for the
// cluster normal at the hit point:
//   SDF_NORMAL_FORWARD_AD -- propagate a Dual3 (value + 3D gradient)
//                            through the full op-tree in a single pass.
//                            ~1.5x the cost of a plain distance eval;
//                            beats central differences (6 evals) once
//                            DE complexity rises. Continuous in space
//                            -- no epsilon-jitter that re-rolls per
//                            bounce, so MIS / SVGF see a clean normal
//                            field.
//   SDF_NORMAL_CENTRAL    -- legacy 6-tap central-difference gradient.
//                            Kept as a fallback for sanity-checking
//                            forward-AD and for ops we haven't ported
//                            yet (none in Phase 2 but the door stays
//                            open for Phase 3 fractal escape-time
//                            normals).
//   SDF_NORMAL_ANALYTIC   -- Phase 1's path. Walks the op-tree and
//                            uses the closed-form per-shape gradient
//                            blended through the smooth-CSG h factor.
//                            Only valid when the cluster has NO
//                            procedural ops (displace-noise / twist
//                            / bend / repeat); the host selects this
//                            automatically for clusters that don't
//                            contain any Phase 2 ops.
static const uint SDF_NORMAL_FORWARD_AD = 0u;
static const uint SDF_NORMAL_CENTRAL    = 1u;
static const uint SDF_NORMAL_ANALYTIC   = 2u;
// --- end SDF Phase 2 ------------------------------------------------------

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
// tree is in `nodes[]`; the host emits nodes in POST-ORDER so the root
// is always the LAST node, at index `node_count - 1` (mirrors the
// host-side convention documented on SdfNode in AnalyticBvh.h, and
// what sdfClusterDist / sdfClusterNormal return below).
// Material+albedo+AABB are in the header. Layout MUST match the SoA
// packing the host writes into the SDF storage buffer (see
// EnsureSdfPrimsUploaded in Engine.cpp).

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
            // k<=0 fallback: opSmoothUnion (distance path) early-outs
            // to min(da,db). The gradient must also be the HARD pick
            // -- the child whose distance won -- otherwise distance
            // and gradient stop corresponding and the surface normal
            // is wrong.
            float kp = n.params.x;
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            if (kp <= 0.0) {
                d = min(da, db);
                g = (da < db) ? grads[n.child_a] : grads[n.child_b];
            } else {
                float h = clamp(0.5 + 0.5 * (db - da) / kp, 0.0, 1.0);
                d = lerp(db, da, h) - kp * h * (1.0 - h);
                g = normalize(lerp(grads[n.child_b], grads[n.child_a], h));
            }
        } else if (n.op == SDF_OP_SMOOTH_SUBTRACT) {
            // k<=0 fallback: opSmoothSubtract -> max(da, -db). Hard
            // gradient is grad_a when (da > -db), -grad_b otherwise.
            float kp = n.params.x;
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            if (kp <= 0.0) {
                d = max(da, -db);
                g = (da > -db) ? grads[n.child_a] : -grads[n.child_b];
            } else {
                float h = clamp(0.5 - 0.5 * (db + da) / kp, 0.0, 1.0);
                d = lerp(da, -db, h) + kp * h * (1.0 - h);
                g = normalize(lerp(grads[n.child_a], -grads[n.child_b], h));
            }
        } else if (n.op == SDF_OP_SMOOTH_INTERSECT) {
            // k<=0 fallback: opSmoothIntersect -> max(da, db).
            float kp = n.params.x;
            float da = dists[n.child_a];
            float db = dists[n.child_b];
            if (kp <= 0.0) {
                d = max(da, db);
                g = (da > db) ? grads[n.child_a] : grads[n.child_b];
            } else {
                float h = clamp(0.5 - 0.5 * (db - da) / kp, 0.0, 1.0);
                d = lerp(db, da, h) + kp * h * (1.0 - h);
                g = normalize(lerp(grads[n.child_b], grads[n.child_a], h));
            }
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

// ----- SDF Phase 2 (#98) procedural / noise / domain ops -------------------
//
// Phase 2 extends the Phase 1 op set with five procedural ops --
// displace-noise, twist, bend, repeat (infinite), repeat-limited
// (bounded). None of these have a closed-form analytic gradient (or
// the closed form is so expensive it's no win), so they go through the
// forward-mode AD path below: Dual3 values carry (scalar, 3-vector
// grad) and every op composes the chain rule on the gradient.
//
// References:
//   * Inigo Quilez, "Distance functions", "Domain repetition", and
//     "Bend / Twist" articles -- our ops match his canon up to the
//     repeat-limited clamp (we use an integer half-extent packed into
//     params[3] rather than free-form clamps so the host wire format
//     stays small).
//   * Karras / Aila, "Forward Mode Automatic Differentiation in PT"
//     -- the gradient propagation pattern.
//
// All distances and parameters in METRES (project mantra). Per-octave
// frequencies are 1 / metres. No demo-scaled tricks anywhere.
//
// COMPILE GATE: the entire block below (noise primitives + warp helpers
// + Dual3 autodiff machinery + procedural cluster walkers + central-diff
// and forward-AD normals) is wrapped in PT_SDF_PROCEDURAL_OPS so default
// builds compile it out. Perf bisect on the default scene attributed a
// ~6 ms/frame regression to this entire block when its mere presence
// got inlined into PathTrace's megakernel sphere-trace dispatch.
#if PT_SDF_PROCEDURAL_OPS

// Hash-based 3D value-noise field with first derivatives. Distinct
// from the cloud-noise hash (PathTraceCloud.slang) so SDF clusters
// don't perturb the cloud noise lookup pattern when both modules are
// imported. Wellons' lowbias32 mixing constants (proven full-spectrum
// output for sequential / lattice integer inputs).
uint sdfHash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float sdfHash3(int3 ip) {
    uint h = sdfHash32(uint(ip.x));
    h = sdfHash32(h ^ uint(ip.y));
    h = sdfHash32(h ^ uint(ip.z));
    return float(h) * (1.0 / 4294967296.0);   // 0..1
}

// 3D value noise with C^1 smoothstep filter. Returns the noise value
// AND its analytic gradient w.r.t. position p (in 1/metres units of
// the input space). Output noise is in [0,1]; gradient is the
// finite-difference-free derivative of the trilinear / cubic mix.
// (See IQ "Value noise derivatives" article for the formula -- this
// is the standard f(x) = 6x^5 - 15x^4 + 10x^3 smoothstep, whose
// derivative is df/dx = 30 x^2 (x-1)^2.)
//
// noise_value = returned scalar; grad in `out_grad` (3 components,
// units of 1/(input metres)). Caller scales by amplitude / frequency
// to chain the rule for FBM and the SDF displace.
float sdfValueNoise3(float3 p, out float3 out_grad) {
    float3 ip = floor(p);
    float3 f  = p - ip;
    int3   i  = int3(ip);

    // Quintic Hermite smoothstep + derivative
    float3 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    float n000 = sdfHash3(i + int3(0,0,0));
    float n100 = sdfHash3(i + int3(1,0,0));
    float n010 = sdfHash3(i + int3(0,1,0));
    float n110 = sdfHash3(i + int3(1,1,0));
    float n001 = sdfHash3(i + int3(0,0,1));
    float n101 = sdfHash3(i + int3(1,0,1));
    float n011 = sdfHash3(i + int3(0,1,1));
    float n111 = sdfHash3(i + int3(1,1,1));

    float k0 = n000;
    float k1 = n100 - n000;
    float k2 = n010 - n000;
    float k3 = n001 - n000;
    float k4 = n000 - n100 - n010 + n110;
    float k5 = n000 - n010 - n001 + n011;
    float k6 = n000 - n100 - n001 + n101;
    float k7 = -n000 + n100 + n010 + n001 - n110 - n011 - n101 + n111;

    float v =
        k0
        + k1 * u.x + k2 * u.y + k3 * u.z
        + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
        + k7 * u.x * u.y * u.z;

    out_grad = float3(
        du.x * (k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z),
        du.y * (k2 + k4 * u.x + k5 * u.z + k7 * u.x * u.z),
        du.z * (k3 + k5 * u.y + k6 * u.x + k7 * u.x * u.y));

    return v;   // 0..1
}

// Fractal Brownian Motion (sum of octaves). Each octave doubles the
// frequency and halves the amplitude. Returns the sum AND its
// gradient -- gradient is essential for the SDF displace path because
// the displace term enters the cluster's value, and forward-mode AD
// needs its derivative to keep the surface normal correct.
//
// FBM is centred in [-1,1] (subtracting 0.5 from value noise then
// scaling by 2) so the displacement term is symmetric: amp * 1.0
// pushes the surface +amp outward in places and -amp inward in
// others, average is zero. This matters because the cluster's AABB
// bound is widened by amp on the host side, and a non-zero-mean
// displacement would shift the surface uniformly out of that bound
// on one face.
float sdfFbm3(float3 p, uint octaves, out float3 out_grad) {
    float  v   = 0.0;
    float3 g   = float3(0.0);
    float  amp = 1.0;
    float  frq = 1.0;
    // Hard cap on octaves at 6 (matches the host's clamp). 6 doublings
    // is the inverse-frequency range from 1 m down to ~3 cm, well
    // below the per-pixel angular extent at sensible camera distances.
    // Dynamic loop -- the unrolled form contributed to a Metal XPC
    // compile timeout when chained through the SDF cluster walker.
    uint o = min(octaves, 6u);
    for (uint i = 0u; i < o; ++i) {
        float3 gn;
        float  vn = sdfValueNoise3(p * frq, gn);
        // Re-centre to [-1, 1]: 2 * (vn - 0.5).
        v += amp * (2.0 * vn - 1.0);
        g += amp * 2.0 * gn * frq;
        amp *= 0.5;
        frq *= 2.0;
    }
    out_grad = g;
    return v;
}

// 3D Worley (cellular) noise. Returns the distance to the closest
// feature point in a 3x3x3 neighbourhood, plus its gradient. Output
// is roughly in [0, 1.5] (cell-diagonal worst case). Useful for
// crater-like or scaly surfaces.
//
// Gradient is the unit vector from the closest cell point toward p,
// which is the standard `(p - feature) / |p - feature|`. At the
// degenerate case |p - feature| == 0 we return zero gradient so the
// caller still gets a sane normal direction from the underlying SDF.
float sdfWorley3(float3 p, out float3 out_grad) {
    float3 ip = floor(p);
    float3 f  = p - ip;
    float  min_d2 = 1e30;
    float3 min_feat = float3(0.0);
    // No `[unroll]` -- the 27-iter triple-nested unroll combined with
    // SDF cluster walker call-sites was a big contributor to the
    // shader's compile-time blow-up. Leave the loop dynamic; the
    // 27-iter bound is small enough that GPU register pressure
    // doesn't suffer.
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int3   cell = int3(ip) + int3(x, y, z);
                float  h1   = sdfHash3(cell);
                float  h2   = sdfHash3(cell + int3(0, 0, 1024));
                float  h3   = sdfHash3(cell + int3(0, 1024, 0));
                float3 feat = float3(x, y, z) + float3(h1, h2, h3) - f;
                float  d2   = dot(feat, feat);
                if (d2 < min_d2) {
                    min_d2   = d2;
                    min_feat = feat;
                }
            }
        }
    }
    float d = sqrt(min_d2);
    out_grad = d > 1e-12 ? (-min_feat / d) : float3(0.0);
    return d;
}

// ----- forward-mode autodiff (Dual3) ---------------------------------------
//
// COMPILE GATE: Dual3 / forward-AD lives behind PT_SDF_AUTODIFF (nested
// inside PT_SDF_PROCEDURAL_OPS). When AUTODIFF is OFF and the user sets
// r_sdf_normal_mode to 0 (forward-AD), the engine logs a [warn] line at
// startup and the trace falls back to central differences.
#if PT_SDF_AUTODIFF
//
// Dual3 carries a scalar value `v` and a 3-vector gradient `d` of v
// w.r.t. world-space position. Each Phase 1 leaf SDF returns a Dual3
// directly via its closed-form gradient (no chain-rule plumbing
// through generic dualXxx helpers -- those generate thread-pointer
// MSL plumbing that bloats the emitted shader and times out Metal's
// XPC compiler on M-class Macs). All SDF distance formulas plus
// their analytic gradients fit in a few lines per shape; we inline
// them into evalLeafDistD below.
//
// The cluster walker (sdfClusterDistD, sdfClusterDistDWarped) then
// composes the LEAF gradients through smooth-CSG ops + procedural
// noise + warp Jacobians with a tiny chain-rule blend at each op
// (one normalize-free 3-vector blend per smin / smax / sdiff).
struct Dual3 {
    float  v;
    float3 d;
};

// ----- Dual3 leaf SDFs (exact closed-form derivatives) ---------------------
//
// `p` is the LEAF-local-frame point (caller subtracts node.center
// before invoking). The gradient is w.r.t. world-space (== leaf-local
// because center translation is a constant). All formulas match
// Phase 1's scalar evalLeafDist / evalLeafNormal exactly so forward-AD
// and Phase 1 analytic-gradient produce identical normals on Phase 1
// clusters.

// sphere: d = |p| - r ; grad = p / |p|
Dual3 sdSphereD(float3 p, float r) {
    Dual3 o;
    float l = length(p);
    o.v = l - r;
    o.d = l > 1e-12 ? p / l : float3(0, 1, 0);
    return o;
}

// box: d = length(max(q,0)) + min(max(q.x,q.y,q.z), 0) ; q = abs(p)-b
// Gradient (outside box): sign(p) * (max(q,0) / |max(q,0)|)
// Gradient (inside box):  sign(p) * unit-along-largest-q-axis
Dual3 sdBoxD(float3 p, float3 b) {
    Dual3 o;
    float3 q  = abs(p) - b;
    float3 sg = sign(p);
    bool outside = max(q.x, max(q.y, q.z)) > 0.0;
    if (outside) {
        float3 m = max(q, float3(0.0));
        float lm = length(m);
        o.v = lm + min(max(q.x, max(q.y, q.z)), 0.0);
        o.d = (lm > 1e-12) ? (sg * (m / lm)) : (sg * float3(0, 1, 0));
    } else {
        o.v = max(q.x, max(q.y, q.z));    // negative inside
        float3 n = float3(0.0);
        if (q.x > q.y && q.x > q.z) n.x = sg.x;
        else if (q.y > q.z)         n.y = sg.y;
        else                        n.z = sg.z;
        o.d = n;
    }
    return o;
}

Dual3 sdRoundBoxD(float3 p, float3 b, float r) {
    Dual3 o = sdBoxD(p, b);
    o.v -= r;   // round-box shrinks the surface inward by r; grad unchanged
    return o;
}

// torus: d = |(|p.xz|-R, p.y)| - r ; grad = (p - closest-on-ring) / |...|
Dual3 sdTorusD(float3 p, float R, float r) {
    Dual3 o;
    float2 pxz = float2(p.x, p.z);
    float  lxz = length(pxz);
    float2 q   = float2(lxz - R, p.y);
    float  lq  = length(q);
    o.v = lq - r;
    if (lxz < 1e-12) {
        o.d = float3(0, sign(p.y), 0);
    } else {
        float2 ringxz = pxz * (R / lxz);
        float3 d      = p - float3(ringxz.x, 0.0, ringxz.y);
        float  ld     = length(d);
        o.d = ld > 1e-12 ? d / ld : float3(0, 1, 0);
    }
    return o;
}

// capsule along Y in [-h, +h] with radius r.
Dual3 sdCapsuleD(float3 p, float h, float r) {
    Dual3 o;
    float  y = clamp(p.y, -h, h);
    float3 q = p - float3(0.0, y, 0.0);
    float  l = length(q);
    o.v = l - r;
    o.d = l > 1e-12 ? q / l : float3(0, 1, 0);
    return o;
}

Dual3 sdPlaneD(float3 p, float3 n, float d) {
    Dual3 o;
    o.v = dot(p, n) + d;
    o.d = n;
    return o;
}

// Dispatch to the right leaf Dual3 SDF.
Dual3 evalLeafDistD(float3 p_local, uint shape, float4 params) {
    if      (shape == SDF_SHAPE_SPHERE)
        return sdSphereD(p_local, params.x);
    else if (shape == SDF_SHAPE_BOX)
        return sdBoxD(p_local, params.xyz);
    else if (shape == SDF_SHAPE_ROUNDED_BOX)
        return sdRoundBoxD(p_local, params.xyz, params.w);
    else if (shape == SDF_SHAPE_TORUS)
        return sdTorusD(p_local, params.x, params.y);
    else if (shape == SDF_SHAPE_CAPSULE)
        return sdCapsuleD(p_local, params.x, params.y);
    else if (shape == SDF_SHAPE_PLANE)
        return sdPlaneD(p_local, params.xyz, params.w);
    Dual3 r;
    r.v = 1e30; r.d = float3(0, 1, 0);
    return r;
}

#endif // PT_SDF_AUTODIFF (Dual3 struct + leaf SDFs + evalLeafDistD)

// ----- procedural op evaluation (scalar distance only) ---------------------
//
// These ops modify the *point* before evaluating the child. The
// distance-only path warps p and reads back the child's distance.
// The forward-AD path below does the same warp but on a Dual3 so the
// gradient comes out chained.

// Twist around the WORLD Y axis: rotates the (x,z) plane by `rate * y`.
// Distance returned is the CHILD's distance evaluated at the warped
// point. Note: the result is *not* a true Lipschitz-1 SDF on the
// twisted shape -- twist stretches space, so the sphere-tracer must
// take smaller-than-distance steps in heavily twisted regions. We
// scale the returned distance by 0.5 to compensate (standard IQ
// trick); this slows the trace but keeps it correct.
//
// PIVOT SEMANTICS: this warp operates on world-space `p` directly,
// so an off-origin child rotates AROUND the world Y axis rather than
// around its own vertical axis. A box centred at (-2, 1, 0) wrapped
// in twist therefore sweeps a wedge of a cylinder around the world
// origin, not a helix in place. The AABB in AnalyticBvh.cpp
// (SDF_OP_TWIST case) takes the world-axis cylinder as its bound to
// keep the slab test conservative under this convention. Users who
// want a local-pivot twist should pre-translate the child to the
// origin, twist, and re-translate via a future translate op (a
// child-local-pivot warp variant is filed as a Phase 3 follow-up).
float3 sdfWarpTwist(float3 p, float rate) {
    float c = cos(rate * p.y);
    float s = sin(rate * p.y);
    return float3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}
// Bend around the WORLD X axis: same world-pivot semantics as
// sdfWarpTwist -- an off-origin child bends around the world X axis
// rather than its own. The AABB code (SDF_OP_BEND case) bounds the
// rotation by a cylinder around the world X axis to stay conservative.
float3 sdfWarpBend(float3 p, float rate) {
    float c = cos(rate * p.x);
    float s = sin(rate * p.x);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}
// Infinite domain repetition with per-axis period. Per-axis period
// of 0 disables repeat on that axis. The returned point is in the
// "tile" local frame: p - period * round(p / period).
float3 sdfWarpRepeat(float3 p, float3 period) {
    float3 q = p;
    if (period.x > 1e-6) q.x = p.x - period.x * round(p.x / period.x);
    if (period.y > 1e-6) q.y = p.y - period.y * round(p.y / period.y);
    if (period.z > 1e-6) q.z = p.z - period.z * round(p.z / period.z);
    return q;
}
// Bounded repetition. `limit` is the per-axis half-extent (integer
// number of tiles either side of origin). cell index is clamped to
// [-limit, +limit].
float3 sdfWarpRepeatLimited(float3 p, float3 period, int3 limit) {
    float3 q = p;
    if (period.x > 1e-6) {
        float idx = clamp(round(p.x / period.x), float(-limit.x), float(limit.x));
        q.x = p.x - period.x * idx;
    }
    if (period.y > 1e-6) {
        float idx = clamp(round(p.y / period.y), float(-limit.y), float(limit.y));
        q.y = p.y - period.y * idx;
    }
    if (period.z > 1e-6) {
        float idx = clamp(round(p.z / period.z), float(-limit.z), float(limit.z));
        q.z = p.z - period.z * idx;
    }
    return q;
}

// Procedural noise displacement helper. Returns the BAND value (-amp
// to +amp, average 0) plus the gradient of that band w.r.t. world p.
// Caller adds the band to the child's distance and accumulates the
// gradient.
float sdfDisplaceNoiseBand(float3 p, float amp, float freq, uint octaves,
                           out float3 out_grad) {
    float3 fbm_grad;
    float  fbm_v = sdfFbm3(p * freq, octaves, fbm_grad);
    // Chain rule: d(amp * fbm(freq * p)) / dp = amp * freq * fbm'(...)
    out_grad = amp * freq * fbm_grad;
    return amp * fbm_v;
}

// Domain-repeat scalar distance helper (NOT AD). Same as Phase 1's
// distance walker for the child node, but at the warped point. Used
// by the scalar sphere-trace's inner loop.
//
// Phase 2's repeat / twist / bend / displace-noise ops all operate
// on a single child sub-tree. The shader's `sdfClusterDist` is
// extended below to handle them in the same flat post-order walk.

// ----- forward-mode AD cluster evaluator -----------------------------------
//
// COMPILE GATE: AD cluster walker + warped variant live behind
// PT_SDF_AUTODIFF (second region, paired with the Dual3 leaves above).
#if PT_SDF_AUTODIFF
//
// Walks the cluster's op-tree once and propagates Dual3 values up to
// the root. Output: .v = signed distance to the surface (same as
// sdfClusterDist), .d = world-space gradient = surface-normal
// direction (unnormalized -- caller normalizes for shading).
//
// Procedural ops are the load-bearing reason for this path: the
// classic 6-tap central-difference approach evaluates the full
// cluster six times per normal query, and DE complexity rises fast
// with procedural noise (FBM with 4 octaves alone is 32 noise
// lookups per sample). Forward-AD is one walk of the tree plus a
// constant-time gradient-vector composition per op.
Dual3 sdfClusterDistDN(SdfCluster c, uint nc, float3 p) {
    // Per-node value / gradient storage as SoA float + float3 arrays
    // rather than `Dual3 vals[N]`. Slang -> MSL emit on the AoS form
    // produces a `thread*` parameter chain through every helper call
    // that times out Metal's XPC compiler; SoA stays as plain register
    // arrays with no thread-pointer plumbing.
    //
    // No `[unroll]` here -- the loop is bounded by node_count which is
    // a small data-dependent count (1..8), and unrolling the 8 cases
    // each containing a 6-way op switch generates ~50 KB of MSL per
    // call site. Loop preservation lets Slang emit a single body with
    // dynamic indexing, which Metal's compiler handles in seconds vs.
    // minutes-and-XPC-timeout on the unrolled form.
    //
    // `nc` is the effective node count (treat nodes [0, nc) as the
    // cluster). Same trick as sdfClusterDistProcN -- avoids the
    // 400-byte SdfCluster struct copy that the warped walker would
    // otherwise need to truncate the tree.
    float  vv[SDF_MAX_NODES];
    float3 vg[SDF_MAX_NODES];
    for (uint i = 0u; i < nc; ++i) {
        SdfClusterNode n = c.nodes[i];
        float  rv;
        float3 rg;
        if (n.op == SDF_OP_LEAF) {
            // Leaf: closed-form distance + gradient via evalLeafDistD
            // (still pass-by-value Dual3 -- single MSL call, tight body).
            Dual3 ld = evalLeafDistD(p - n.center, n.shape, n.params);
            rv = ld.v; rg = ld.d;
        } else if (n.op == SDF_OP_SMOOTH_UNION) {
            float kp = n.params.x;
            float  av = vv[n.child_a]; float3 ag = vg[n.child_a];
            float  bv = vv[n.child_b]; float3 bg = vg[n.child_b];
            if (kp <= 0.0) {
                rv = (av < bv) ? av : bv;
                rg = (av < bv) ? ag : bg;
            } else {
                float h = clamp(0.5 + 0.5 * (bv - av) / kp, 0.0, 1.0);
                rv = lerp(bv, av, h) - kp * h * (1.0 - h);
                rg = lerp(bg, ag, h);
            }
        } else if (n.op == SDF_OP_SMOOTH_SUBTRACT) {
            float kp = n.params.x;
            float  av = vv[n.child_a]; float3 ag = vg[n.child_a];
            float  bv = vv[n.child_b]; float3 bg = vg[n.child_b];
            if (kp <= 0.0) {
                rv = max(av, -bv);
                rg = (av > -bv) ? ag : -bg;
            } else {
                float h = clamp(0.5 - 0.5 * (bv + av) / kp, 0.0, 1.0);
                rv = lerp(av, -bv, h) + kp * h * (1.0 - h);
                rg = lerp(ag, -bg, h);
            }
        } else if (n.op == SDF_OP_SMOOTH_INTERSECT) {
            float kp = n.params.x;
            float  av = vv[n.child_a]; float3 ag = vg[n.child_a];
            float  bv = vv[n.child_b]; float3 bg = vg[n.child_b];
            if (kp <= 0.0) {
                rv = (av > bv) ? av : bv;
                rg = (av > bv) ? ag : bg;
            } else {
                float h = clamp(0.5 - 0.5 * (bv - av) / kp, 0.0, 1.0);
                rv = lerp(bv, av, h) + kp * h * (1.0 - h);
                rg = lerp(bg, ag, h);
            }
        } else if (n.op == SDF_OP_DISPLACE) {
            // Constant displace: child value bumped by params[0];
            // gradient unchanged.
            rv = vv[n.child_a] + n.params.x;
            rg = vg[n.child_a];
        } else if (n.op == SDF_OP_DISPLACE_NOISE) {
            // Noise displace: child + amp * fbm(freq * p); add the
            // analytic noise gradient to the child gradient.
            float3 noise_grad;
            float  band = sdfDisplaceNoiseBand(p, n.params.x, n.params.y,
                                               asuint(n.params.z), noise_grad);
            rv = vv[n.child_a] + band;
            rg = vg[n.child_a] + noise_grad;
        } else {
            // Warp ops (twist / bend / repeat / repeat-limited) are
            // handled at the outer warped walker (sdfClusterDistDWarped)
            // -- they should never appear inside this inner walk. The
            // host's wrap_proc rejects nested warps up-front (see the
            // "warps must be root" invariant comment there). If one
            // slips through anyway (host bug or hand-crafted cluster),
            // FAIL VISIBLY rather than silently dropping the warp: emit
            // the +inf sentinel so the cluster surface vanishes from
            // the trace -- much easier to spot than a subtly-wrong
            // shape that lacks the warp it was supposed to have.
            rv = 1e30;
            rg = float3(0, 1, 0);
        }
        vv[i] = rv;
        vg[i] = rg;
    }
    Dual3 r;
    if (nc > 0u) {
        r.v = vv[nc - 1u];
        r.d = vg[nc - 1u];
    } else {
        r.v = 1e30;
        r.d = float3(0, 1, 0);
    }
    return r;
}

// Convenience wrapper that walks the full cluster's tree. Used by
// call sites that don't need the sub-tree truncation trick that the
// warped walker uses below.
Dual3 sdfClusterDistD(SdfCluster c, float3 p) {
    return sdfClusterDistDN(c, c.node_count, p);
}

// Warped forward-mode AD. When a cluster's root is a space-warp op
// (twist / bend / repeat / repeat-limited), we warp the input point
// once, walk the entire op-tree at the warped p, and then apply the
// warp's Jacobian to the resulting gradient. For Phase 2 we ship
// twist / bend / repeat / repeat-limited as "single-warp" wrappers:
// the host enforces the warp is the root and its child is itself a
// leaf or a chain that doesn't contain another warp. That keeps the
// chain rule a single Jacobian, not a stack.
//
// Twist Jacobian (rotation about Y by theta = rate * y):
//   d/dx q.x =  cos(theta)   d/dy q.x = -rate*sin(theta)*x - rate*cos(theta)*z
//   d/dz q.x = -sin(theta)
//   d/dx q.z =  sin(theta)   d/dy q.z =  rate*cos(theta)*x - rate*sin(theta)*z
//   d/dz q.z =  cos(theta)
//   q.y = p.y (identity column on Y)
//
// We multiply (J^T * grad) to map the warped-space gradient back
// to world space.
Dual3 sdfClusterDistDWarped(SdfCluster c, float3 p) {
    // Look at the root op to decide which warp to apply. The warp
    // child sub-tree is at indices [0, rn.child_a + 1) by host
    // post-order convention -- we pass that as `nc` to
    // sdfClusterDistDN so we walk only the child, without copying
    // the cluster struct.
    uint root = c.node_count > 0u ? c.node_count - 1u : 0u;
    SdfClusterNode rn = c.nodes[root];
    if (rn.op == SDF_OP_TWIST) {
        float rate = rn.params.x;
        float3 wp = sdfWarpTwist(p, rate);
        float th = rate * p.y;
        float ct = cos(th), st = sin(th);
        Dual3 r = sdfClusterDistDN(c, rn.child_a + 1u, wp);
        // Jacobian transpose:
        //   q = (ct*x - st*z, y, st*x + ct*z)
        //   dq/dp = [[ ct, -rate*(st*x + ct*z), -st ],
        //            [  0,                  1,    0 ],
        //            [ st,  rate*(ct*x - st*z),  ct ]]
        float gx = ct * r.d.x + st * r.d.z;
        float gy = -rate * (st * p.x + ct * p.z) * r.d.x + r.d.y
                 +  rate * (ct * p.x - st * p.z) * r.d.z;
        float gz = -st * r.d.x + ct * r.d.z;
        r.d = float3(gx, gy, gz);
        // Lipschitz fudge: twist isn't a 1-Lipschitz SDF; scale
        // distance down so the sphere-tracer doesn't overshoot. The
        // exact safe bound is 1 / (1 + |rate| * r_max) where r_max is
        // the child's maximum distance from the twist axis (Inigo
        // Quilez's standard derivation). 0.5 is correct for any
        // |rate| * r_max <= 1; the Phase 2 fixture (rate=1.8/m,
        // r_max ~= 0.42 m -> product 0.76) stays within that bound,
        // and the sphere tracer's per-step epsilon clamp absorbs the
        // slight overshoot in heavily-twisted regions. A
        // rate-and-extent-aware scale is filed as a Phase 3 follow-up
        // for very-high-rate / very-off-axis twists.
        r.v *= 0.5;
        return r;
    } else if (rn.op == SDF_OP_BEND) {
        float rate = rn.params.x;
        float3 wp = sdfWarpBend(p, rate);
        float th = rate * p.x;
        float ct = cos(th), st = sin(th);
        Dual3 r = sdfClusterDistDN(c, rn.child_a + 1u, wp);
        // Bend Jacobian (rotation about X by theta = rate * x):
        //   q.x = x, q.y = ct*y - st*z, q.z = st*y + ct*z
        // J^T * grad:
        float gx = r.d.x
                 + (-rate * (st * p.y + ct * p.z)) * r.d.y
                 + ( rate * (ct * p.y - st * p.z)) * r.d.z;
        float gy = ct * r.d.y + st * r.d.z;
        float gz = -st * r.d.y + ct * r.d.z;
        r.d = float3(gx, gy, gz);
        // Same 0.5 Lipschitz fudge as TWIST -- correct for any
        // |rate| * r_max <= 1 where r_max is the child's maximum
        // distance from the bend (X) axis. Phase 2 fixture stays
        // safely under that bound; a rate-aware scale is a Phase 3
        // follow-up.
        r.v *= 0.5;
        return r;
    } else if (rn.op == SDF_OP_REPEAT) {
        float3 wp = sdfWarpRepeat(p, rn.params.xyz);
        // Domain-repeat Jacobian is the identity ALMOST everywhere
        // (round() is locally constant -- the only discontinuities
        // are at the cell boundaries where round changes value).
        return sdfClusterDistDN(c, rn.child_a + 1u, wp);
    } else if (rn.op == SDF_OP_REPEAT_LIMITED) {
        uint   packed = asuint(rn.params.w);
        int3   lim = int3(
            int( packed         & 0x3ffu),
            int((packed >> 10) & 0x3ffu),
            int((packed >> 20) & 0x3ffu));
        float3 wp = sdfWarpRepeatLimited(p, rn.params.xyz, lim);
        return sdfClusterDistDN(c, rn.child_a + 1u, wp);
    }
    // No warp at root: fall back to the plain AD walk.
    return sdfClusterDistDN(c, c.node_count, p);
}

#endif // PT_SDF_AUTODIFF (AD cluster walker + warped variant)

// Scalar distance walker that also handles SDF_OP_DISPLACE_NOISE.
// Used by `sdfClusterDistWarped` (warp-aware outer wrapper below)
// and by `sdfClusterTrace`'s inner step loop when the cluster has
// no warp ops at the root. Duplicates the Phase 1 walker with the
// additional displace-noise case (twist / bend / repeat are handled
// at the warped walker layer above, not in this inner walker).
//
// `nc` is the EFFECTIVE node count -- the walker treats nodes
// [0, nc) as the cluster, ignoring everything from nc onward. The
// warped walker passes `c.nodes[root].child_a + 1` here to walk
// the warp's CHILD sub-tree without copying the SdfCluster struct
// (which is ~400 bytes and was producing per-call-site MSL struct
// copies that bloated Metal's compile time).
float sdfClusterDistProcN(SdfCluster c, uint nc, float3 p) {
    float dists[SDF_MAX_NODES];
    for (uint i = 0u; i < nc; ++i) {
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
        } else if (n.op == SDF_OP_DISPLACE_NOISE) {
            float3 nullGrad;
            float  band = sdfDisplaceNoiseBand(p, n.params.x, n.params.y,
                                               asuint(n.params.z), nullGrad);
            d = dists[n.child_a] + band;
        } else {
            // Warp ops shouldn't be evaluated by the inner walker;
            // they're handled by sdfClusterDistWarped at the root.
            // The host's wrap_proc rejects nested warps up-front (see
            // the "warps must be root" invariant on the host side), so
            // a non-root warp here means an invalid cluster slipped
            // past validation. FAIL VISIBLY (+inf sentinel) so the
            // cluster surface disappears from the trace -- much
            // clearer feedback than silently dropping the warp and
            // shipping a subtly-wrong shape.
            d = 1e30;
        }
        dists[i] = d;
    }
    return nc > 0u ? dists[nc - 1u] : 1e30;
}

// Convenience wrapper around sdfClusterDistProcN that walks the full
// cluster (all node_count nodes). Used by call sites that don't need
// the sub-tree truncation trick.
float sdfClusterDistProc(SdfCluster c, float3 p) {
    return sdfClusterDistProcN(c, c.node_count, p);
}

// Scalar (non-AD) variant of the warped walker. Used by the sphere
// tracer's step distance -- we don't need the gradient at every
// step, only the distance. Cheaper than sdfClusterDistDWarped by
// the 3-vector grad propagation cost.
//
// NOTE: passes a TRUNCATED node count to sdfClusterDistProcN so the
// child sub-tree gets walked without copying the SdfCluster struct.
// The cluster is ~400 bytes; pre-truncation we did `SdfCluster cc =
// c; cc.node_count = ...; sdfClusterDistProc(cc, wp);` which forced
// Slang/Metal to emit four 400-byte struct copies per call (one per
// warp branch). Metal's compiler took ~minutes on that pattern,
// passing the `nc` argument keeps it to seconds.
float sdfClusterDistWarped(SdfCluster c, float3 p) {
    uint root = c.node_count > 0u ? c.node_count - 1u : 0u;
    SdfClusterNode rn = c.nodes[root];
    if (rn.op == SDF_OP_TWIST) {
        float3 wp = sdfWarpTwist(p, rn.params.x);
        // 0.5 Lipschitz fudge -- mirrors the AD walker's TWIST branch.
        // Safe for |rate| * r_max <= 1; a rate-aware bound is a
        // Phase 3 follow-up.
        return 0.5 * sdfClusterDistProcN(c, rn.child_a + 1u, wp);
    } else if (rn.op == SDF_OP_BEND) {
        float3 wp = sdfWarpBend(p, rn.params.x);
        // Same 0.5 Lipschitz fudge as TWIST -- see the AD walker's
        // BEND branch for the |rate| * r_max <= 1 safety condition.
        return 0.5 * sdfClusterDistProcN(c, rn.child_a + 1u, wp);
    } else if (rn.op == SDF_OP_REPEAT) {
        float3 wp = sdfWarpRepeat(p, rn.params.xyz);
        return sdfClusterDistProcN(c, rn.child_a + 1u, wp);
    } else if (rn.op == SDF_OP_REPEAT_LIMITED) {
        uint   packed = asuint(rn.params.w);
        int3   lim = int3(
            int( packed         & 0x3ffu),
            int((packed >> 10) & 0x3ffu),
            int((packed >> 20) & 0x3ffu));
        float3 wp = sdfWarpRepeatLimited(p, rn.params.xyz, lim);
        return sdfClusterDistProcN(c, rn.child_a + 1u, wp);
    }
    return sdfClusterDistProcN(c, c.node_count, p);
}

// True if this cluster contains any Phase 2 procedural op (and thus
// the analytic-gradient walker in Phase 1 can't compute its normal
// correctly). The host pre-computes this and the shader could too,
// but we already have to walk the op array for the distance eval so
// a per-frame check is cheap.
bool sdfClusterHasProcedural(SdfCluster c) {
    for (uint i = 0u; i < c.node_count; ++i) {
        uint op = c.nodes[i].op;
        if (op == SDF_OP_DISPLACE_NOISE ||
            op == SDF_OP_TWIST ||
            op == SDF_OP_BEND ||
            op == SDF_OP_REPEAT ||
            op == SDF_OP_REPEAT_LIMITED) return true;
    }
    return false;
}

// Central-difference normal -- the fallback when forward-AD isn't
// applicable or when the host's r_sdf_normal_mode cvar requests it.
// h is the per-axis epsilon; 1e-4 m is a sensible default (matches
// r_sdf_epsilon's surface-distance terminator). 6 evals: x+/x-/y+/y-
// /z+/z- (Müller's "tetrahedron technique" would be 4 evals but
// 6-tap central is more accurate at the same surface offset and
// trivially easier to read; the cost difference is one cluster-walk
// per ray-pixel so per-frame overhead is negligible vs. forward-AD's
// 1-walk wins on procedural-heavy clusters).
float3 sdfClusterNormalCentral(SdfCluster c, float3 p, float h) {
    float dx_p = sdfClusterDistWarped(c, p + float3(h, 0, 0));
    float dx_n = sdfClusterDistWarped(c, p - float3(h, 0, 0));
    float dy_p = sdfClusterDistWarped(c, p + float3(0, h, 0));
    float dy_n = sdfClusterDistWarped(c, p - float3(0, h, 0));
    float dz_p = sdfClusterDistWarped(c, p + float3(0, 0, h));
    float dz_n = sdfClusterDistWarped(c, p - float3(0, 0, h));
    float3 g = float3(dx_p - dx_n, dy_p - dy_n, dz_p - dz_n);
    float l = length(g);
    return l > 1e-12 ? g / l : float3(0, 1, 0);
}

// Forward-mode AD normal. Cheapest reliable normal for clusters with
// procedural ops -- one walk of the op-tree, gradient comes out in
// the resulting Dual3.d. Caller normalizes.
//
// COMPILE GATE: PT_SDF_AUTODIFF (third nested region). When OFF the
// trace falls back to sdfClusterNormalCentral for procedural clusters
// regardless of the r_sdf_normal_mode cvar.
#if PT_SDF_AUTODIFF
float3 sdfClusterNormalForwardAD(SdfCluster c, float3 p) {
    Dual3 r = sdfClusterDistDWarped(c, p);
    float l = length(r.d);
    return l > 1e-12 ? r.d / l : float3(0, 1, 0);
}
#endif // PT_SDF_AUTODIFF (forward-AD normal)

#endif // PT_SDF_PROCEDURAL_OPS (Phase 2 procedural ops + walkers + normals)

// ----- tri-planar texture projection (#74 binding point) -------------------
//
// SDFs have no inherent UV. Tri-planar projects the world-space hit
// point onto the three coordinate planes (XY, YZ, ZX), samples a 2D
// texture on each plane, and blends by the surface-normal alignment
// to each plane's normal. Sharp edges of the blend (power-blend
// exponent) hide the seams.
//
// `sample_xy(uv)` etc. are caller-supplied texture sampling lambdas
// in HLSL/Slang; here we just compute the WEIGHTED ARGUMENT BUNDLE
// (the three sets of (uv, weight) pairs) so #74's material code can
// pull in whatever sampler it likes. Returning the bundle keeps this
// module free of texture / sampler resources (those are owned by
// PathTrace.slang and the material module).
//
// Usage:
//   TriplanarSample tp = sdfTriplanarSample(p_world, normal_world, scale);
//   float3 color = tp.w.x * sample(tp.uv_yz, ...)
//                + tp.w.y * sample(tp.uv_zx, ...)
//                + tp.w.z * sample(tp.uv_xy, ...);
struct TriplanarSample {
    float2 uv_yz;     // sampled by faces whose normal points +/-X
    float2 uv_zx;     // sampled by faces whose normal points +/-Y
    float2 uv_xy;     // sampled by faces whose normal points +/-Z
    float3 w;         // power-blend weights, sums to 1
};
TriplanarSample sdfTriplanarSample(float3 p_world, float3 normal_world, float scale) {
    TriplanarSample r;
    // Project onto each axis-aligned plane. scale is metres-per-tile;
    // smaller = denser tiling.
    float s = max(scale, 1e-4);
    r.uv_yz = float2(p_world.y, p_world.z) / s;
    r.uv_zx = float2(p_world.z, p_world.x) / s;
    r.uv_xy = float2(p_world.x, p_world.y) / s;
    // Power blend with exponent 4 (visible silhouette stays clean,
    // axis-crossings get a soft fade ~5 deg wide). Take abs of the
    // normal so the blend is symmetric front/back along each axis.
    float3 wn = pow(abs(normal_world), float3(4.0));
    float  sum = wn.x + wn.y + wn.z;
    r.w = sum > 1e-6 ? wn / sum : float3(1.0/3.0);
    return r;
}
// --- end SDF Phase 2 (#98) procedural / noise / domain ops -----------------

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
// (default 1e-4 metres). normal_mode picks between the SDF_NORMAL_*
// constants above and is sourced from the host cvar
// `r_sdf_normal_mode` (default 0 = forward-AD).
bool sdfClusterTrace(SdfCluster c, float3 ro, float3 rd,
                     float t_min, float t_max,
                     uint max_iters, float epsilon,
                     uint normal_mode,
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

#if PT_SDF_PROCEDURAL_OPS
    // Phase 2: clusters containing procedural ops (noise displace,
    // twist, bend, repeat, repeat-limited) need the Phase 2 walker
    // for both distance (warp-aware) and normal (forward-AD or
    // central-difference). Phase 1 clusters keep the analytic-
    // gradient path for backward-compat goldens.
    //
    // Detection: walk the op array once at trace entry, set the
    // `procedural` flag. The trace inner loop and the normal eval
    // both branch on this flag so non-procedural clusters never pay
    // the Phase 2 walker overhead.
    bool procedural = sdfClusterHasProcedural(c);
#endif

    // Step bound: cap max_iters at SDF_MAX_TRACE_ITERS to keep an upper
    // bound on the shader's worst-case work. 256 is a generous Phase 1
    // ceiling for analytic primitives + smooth ops (fractals in Phase 3
    // will need more).
    //
    // Inside-surface guard: a ray that starts inside the SDF (e.g. a
    // dielectric refraction ray spawned a tiny step into the medium)
    // sees d < 0 on its first sample. The naive `if (d < epsilon) hit`
    // + `t += d` would (a) report an immediate self-hit at t = t_min,
    // and (b) step BACKWARDS along the ray, never escaping. Use
    // |d| for both the hit test and the advance: that's still a
    // distance-bounded step (we never tunnel through a thin surface),
    // and it lets the trace march forward until it crosses out through
    // the surface from the inside. Note `epsilon` is intentionally
    // small enough (default 1e-4 m) that we don't accept the start
    // point itself as a hit unless we genuinely are within `epsilon`
    // of the surface to begin with.
    uint iters = min(max_iters, 256u);
    for (uint i = 0u; i < iters; ++i) {
        float3 p = ro + rd * t;
        float  d;
#if PT_SDF_PROCEDURAL_OPS
        // Procedural cluster -> warp-aware walker so displace-noise /
        // twist / bend / repeat all contribute to the distance. The
        // Phase 1 path stays the cheap analytic walker for legacy
        // (non-procedural) clusters.
        d = procedural ? sdfClusterDistWarped(c, p) : sdfClusterDist(c, p);
#else
        // Procedural ops compiled out: the only walker available is
        // the analytic-CSG one. A cluster uploaded with a Phase 2 op
        // id falls into sdfClusterDist's `else { d = 1e30; }` branch
        // and the trace silently skips it -- correct behaviour for
        // a build that opted out of procedural support.
        d = sdfClusterDist(c, p);
#endif
        float abs_d = abs(d);
        if (abs_d < epsilon) {
            out_t = t;
            // Normal source:
            //   procedural + forward-AD     -> sdfClusterNormalForwardAD
            //   procedural + central-diff   -> sdfClusterNormalCentral
            //   non-procedural (Phase 1)    -> sdfClusterNormal
            //     (closed-form analytic; same path as Phase 1
            //      golden image so its bit-exactness is preserved.
            //      A future ANALYTIC selector slot stays open via
            //      SDF_NORMAL_ANALYTIC even though the dispatcher
            //      currently routes "non-procedural" to the same
            //      path -- the constant exists so per-cluster
            //      override stays cheap to wire on later.)
#if PT_SDF_PROCEDURAL_OPS
            if (procedural) {
#  if PT_SDF_AUTODIFF
                if (normal_mode == SDF_NORMAL_CENTRAL) {
                    out_normal = sdfClusterNormalCentral(c, p, epsilon);
                } else {
                    out_normal = sdfClusterNormalForwardAD(c, p);
                }
#  else
                // AD path compiled out -> always central differences.
                // The engine emits a [warn] at startup if
                // r_sdf_normal_mode is 0 (forward-AD) but this build
                // doesn't include the AD walker; the cvar continues
                // to read 0 but the trace silently uses central diff
                // here so a procedural scene still renders.
                out_normal = sdfClusterNormalCentral(c, p, epsilon);
#  endif
            } else {
                out_normal = normalize(sdfClusterNormal(c, p));
            }
#else
            // Procedural ops compiled out: Phase 1 analytic normal
            // is the only path. normal_mode is unused in this build.
            out_normal = normalize(sdfClusterNormal(c, p));
#endif
            return true;
        }
        // Always advance by a strictly-positive step. Floor at epsilon
        // so a ray that's exactly on the surface (|d| ~ 0 from the
        // previous step's overshoot) still makes progress.
        t += max(abs_d, epsilon);
        if (t >= t_end) return false;
    }
    return false;
}

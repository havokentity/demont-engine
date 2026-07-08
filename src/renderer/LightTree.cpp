// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte

#include "renderer/LightTree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace pt::renderer {

// Type-tag mirrors of Engine::AnalyticLight::Type. The renderer lib
// doesn't depend on the engine, so we keep our own enumeration of the
// same integer values (Engine::EnsureLightTreeUploaded passes through
// the int form via LightInput::type).
namespace {

constexpr std::uint32_t kLightPoint  = 0;
constexpr std::uint32_t kLightSpot   = 1;
constexpr std::uint32_t kLightSphere = 2;
constexpr std::uint32_t kLightQuad   = 3;

// Per-light summary used during the build. Each summary is reduced
// (combined) into the cluster summary held on each tree node. The
// reduction operators match Conty Estevez & Kulla 2018, section 4.2:
//   - Position AABB: pairwise union of child AABBs.
//   - Intensity: scalar sum (we keep an RGB-averaged luminance so a
//     warm light vs. a cold one of the same wattage rank correctly).
//   - Emission cone: pairwise cone union -- the smallest cone that
//     bounds both child cones. Closed-form when both axes are unit
//     vectors and the cones lie in 3D; see CombineCones below.
struct LightSummary {
    float aabb_min[3]    {0, 0, 0};
    float aabb_max[3]    {0, 0, 0};
    float cone_axis[3]   {0, 0, 1};   // unit; valid even for "spherical" emitters
    float cone_cos_half  {1.0f};      // 1 = single direction, -1 = sphere
    float intensity      {0.0f};      // scalar luminance
    std::uint32_t light_id {0};       // index into the input vector
};

// Scalar-luminance approximation. We use the standard Rec.709 weights;
// the importance heuristic only needs a relative magnitude per cluster
// so the exact weights don't matter as long as siblings agree.
inline float ScalarLuminance(const float rgb[3]) {
    return 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
}

// Per-type point-bound + cone derivation. The position AABB is the
// emitter's *bounding extent* (not the position alone) so the tree's
// AABB tells the shader's distance² estimate "if you're outside this
// box, all the emitters in this cluster are at least X away" -- which
// is exactly the geometric falloff the importance heuristic wants.
//
// Cone of effective emission directions:
//   - POINT:  fully spherical (axis irrelevant; cos_half = -1).
//   - SPOT:   axis = spot direction; cos_half = cos_outer.
//   - SPHERE: fully spherical (radiates in all surface directions).
//   - QUAD:   one-sided emitter; half-cone is the hemisphere (cos_half = 0)
//             centred on the quad normal.
LightSummary MakeLeafSummary(const LightInput& L,
                              std::uint32_t light_id) {
    LightSummary s;
    s.light_id = light_id;

    // Position AABB: a tiny ε box around the position for point/spot,
    // a radius-sized box for sphere, and a u/v-half-extent-sized box
    // for quad. The shader's cluster-distance estimate inflates the
    // box; over-tight bounds hurt accuracy more than over-loose ones.
    if (L.type == kLightPoint ||
        L.type == kLightSpot) {
        constexpr float eps = 1e-3f;  // 1mm box around the position
        for (int i = 0; i < 3; ++i) {
            s.aabb_min[i] = L.pos[i] - eps;
            s.aabb_max[i] = L.pos[i] + eps;
        }
    } else if (L.type == kLightSphere) {
        const float r = std::max(L.radius, 1e-3f);
        for (int i = 0; i < 3; ++i) {
            s.aabb_min[i] = L.pos[i] - r;
            s.aabb_max[i] = L.pos[i] + r;
        }
    } else {  // Quad
        // Quad u-vector has length = u_half; v-axis is the cross product
        // of normal and u, with length v_half. Bound the full rect by
        // taking |u_x| + |v_axis_x| extents on each axis.
        const float ux = std::fabs(L.u_vec[0]);
        const float uy = std::fabs(L.u_vec[1]);
        const float uz = std::fabs(L.u_vec[2]);
        // Reconstruct v-axis = normalize(cross(L.dir, L.u_vec)) * L.v_half.
        // For tree-build purposes we just need extent magnitudes;
        // |L.dir|*|L.u_vec| / |L.dir| = |L.u_vec|, then scale by v_half.
        // Since dir is unit, cross magnitude on each axis is bounded by
        // |L.u_vec|, so v-extent <= L.v_half. Conservative: use v_half
        // uniformly in each component.
        const float vh = std::fabs(L.v_half);
        for (int i = 0; i < 3; ++i) {
            const float u_ext = (i == 0) ? ux : (i == 1) ? uy : uz;
            const float ext   = u_ext + vh;
            s.aabb_min[i] = L.pos[i] - ext;
            s.aabb_max[i] = L.pos[i] + ext;
        }
    }

    // Cone of effective emission directions.
    if (L.type == kLightPoint ||
        L.type == kLightSphere) {
        // Isotropic: cos_half = -1 means "any axis is valid"; we stash
        // +Y as a placeholder axis so the unit-vector invariant holds.
        s.cone_axis[0] = 0; s.cone_axis[1] = 1; s.cone_axis[2] = 0;
        s.cone_cos_half = -1.0f;
    } else if (L.type == kLightSpot) {
        // Spot's L.dir is unit; cos_outer is the outer half-angle.
        s.cone_axis[0] = L.dir[0];
        s.cone_axis[1] = L.dir[1];
        s.cone_axis[2] = L.dir[2];
        s.cone_cos_half = L.cos_outer;
    } else {  // Quad
        s.cone_axis[0] = L.dir[0];
        s.cone_axis[1] = L.dir[1];
        s.cone_axis[2] = L.dir[2];
        // One-sided emitter: hemisphere about the normal.
        s.cone_cos_half = 0.0f;
    }

    // Intensity: scalar luminance. For area emitters (sphere/quad) we
    // multiply by surface area so a big dim emitter ranks against a
    // small bright one by total emitted power (the cluster importance
    // metric the path tracer evaluates).
    float lum = ScalarLuminance(L.intensity);
    if (L.type == kLightSphere) {
        const float r = std::max(L.radius, 1e-3f);
        const float area = 4.0f * 3.14159265358979f * r * r;
        lum *= area;
    } else if (L.type == kLightQuad) {
        const float u_len = std::sqrt(L.u_vec[0] * L.u_vec[0] +
                                       L.u_vec[1] * L.u_vec[1] +
                                       L.u_vec[2] * L.u_vec[2]);
        const float area = 4.0f * u_len * std::fabs(L.v_half);
        lum *= area;
    }
    s.intensity = lum;

    return s;
}

// Pairwise AABB union.
inline void UnionAabb(float out_min[3], float out_max[3],
                       const float a_min[3], const float a_max[3],
                       const float b_min[3], const float b_max[3]) {
    for (int i = 0; i < 3; ++i) {
        out_min[i] = std::min(a_min[i], b_min[i]);
        out_max[i] = std::max(a_max[i], b_max[i]);
    }
}

// Pairwise cone union. Conty Estevez & Kulla 2018, eq. 1: given two
// cones (axis_a, theta_a) and (axis_b, theta_b), find the smallest cone
// (axis_c, theta_c) that contains both. Closed form:
//   theta_d = angle between axes
//   If theta_d + theta_b <= theta_a  -> child b sits inside a; result = a.
//   If theta_d + theta_a <= theta_b  -> child a sits inside b; result = b.
//   Else theta_c = (theta_a + theta_b + theta_d) / 2 and axis_c is
//   rotated from axis_a by (theta_c - theta_a) toward axis_b.
//
// We work in cos-half-angle space directly: cones are stored as the
// pair (axis, cos_half_angle). The "fully spherical" sentinel
// cos_half = -1 short-circuits to "result is spherical".
void CombineCones(const float axis_a[3], float cos_half_a,
                   const float axis_b[3], float cos_half_b,
                   float out_axis[3], float& out_cos_half) {
    // Either side fully spherical -> result fully spherical.
    if (cos_half_a <= -0.999f || cos_half_b <= -0.999f) {
        // Pick whichever side had the more meaningful axis just in case
        // a future caller cares about the placeholder. Cos = -1 marks
        // the spherical case unambiguously regardless.
        out_axis[0] = axis_a[0];
        out_axis[1] = axis_a[1];
        out_axis[2] = axis_a[2];
        out_cos_half = -1.0f;
        return;
    }

    // theta_d = angle between axes.
    float dot_ab = axis_a[0] * axis_b[0]
                 + axis_a[1] * axis_b[1]
                 + axis_a[2] * axis_b[2];
    if (dot_ab >  1.0f) dot_ab =  1.0f;
    if (dot_ab < -1.0f) dot_ab = -1.0f;
    const float theta_d = std::acos(dot_ab);

    // Convert cos_half to half-angle in radians.
    const float theta_a = std::acos(std::clamp(cos_half_a, -1.0f, 1.0f));
    const float theta_b = std::acos(std::clamp(cos_half_b, -1.0f, 1.0f));

    // Containment short-circuits.
    if (theta_d + theta_b <= theta_a) {
        out_axis[0] = axis_a[0];
        out_axis[1] = axis_a[1];
        out_axis[2] = axis_a[2];
        out_cos_half = cos_half_a;
        return;
    }
    if (theta_d + theta_a <= theta_b) {
        out_axis[0] = axis_b[0];
        out_axis[1] = axis_b[1];
        out_axis[2] = axis_b[2];
        out_cos_half = cos_half_b;
        return;
    }

    // Union cone half-angle.
    float theta_c = 0.5f * (theta_a + theta_b + theta_d);
    if (theta_c >= 3.14159265358979f) {
        // Cone exceeds a hemisphere by enough that "fully spherical"
        // is the cheapest representation. Keep the axis we have.
        out_axis[0] = axis_a[0];
        out_axis[1] = axis_a[1];
        out_axis[2] = axis_a[2];
        out_cos_half = -1.0f;
        return;
    }
    out_cos_half = std::cos(theta_c);

    // Rotate axis_a toward axis_b by (theta_c - theta_a). Use the
    // perpendicular component of axis_b minus its projection on axis_a.
    const float rot = theta_c - theta_a;
    const float cos_r = std::cos(rot);
    const float sin_r = std::sin(rot);
    float perp[3] = {
        axis_b[0] - dot_ab * axis_a[0],
        axis_b[1] - dot_ab * axis_a[1],
        axis_b[2] - dot_ab * axis_a[2],
    };
    const float perp_len = std::sqrt(perp[0] * perp[0]
                                       + perp[1] * perp[1]
                                       + perp[2] * perp[2]);
    if (perp_len > 1e-6f) {
        const float inv = 1.0f / perp_len;
        perp[0] *= inv; perp[1] *= inv; perp[2] *= inv;
    } else {
        // Axes collinear -- rotation direction undefined; use as-is.
        out_axis[0] = axis_a[0];
        out_axis[1] = axis_a[1];
        out_axis[2] = axis_a[2];
        return;
    }
    out_axis[0] = cos_r * axis_a[0] + sin_r * perp[0];
    out_axis[1] = cos_r * axis_a[1] + sin_r * perp[1];
    out_axis[2] = cos_r * axis_a[2] + sin_r * perp[2];
    // Renormalise to defend against drift.
    const float ax_len = std::sqrt(out_axis[0] * out_axis[0]
                                     + out_axis[1] * out_axis[1]
                                     + out_axis[2] * out_axis[2]);
    if (ax_len > 1e-6f) {
        const float inv = 1.0f / ax_len;
        out_axis[0] *= inv;
        out_axis[1] *= inv;
        out_axis[2] *= inv;
    }
}

// (CombineSummaries was deleted: the PR #196 in-place rebuild folds
// child cluster data directly at the bottom of BuildRecursive, which
// is the single authoritative reduction -- a stale copy here invited
// edits that silently changed nothing in the shipped tree.)

// Write a leaf node IN PLACE at out_nodes[dst_idx]. Caller is
// responsible for having sized out_nodes so dst_idx is valid.
inline void WriteLeafAt(const LightSummary& s,
                        std::vector<LightTreeNode>& out_nodes,
                        std::uint32_t dst_idx) {
    LightTreeNode& node = out_nodes[dst_idx];
    for (int i = 0; i < 3; ++i) {
        node.aabb_min[i] = s.aabb_min[i];
        node.aabb_max[i] = s.aabb_max[i];
        node.cone_axis[i] = s.cone_axis[i];
    }
    node.cone_cos_half = s.cone_cos_half;
    node.intensity     = s.intensity;
    node.left_first    = s.light_id;
    node.count         = 1u;
}

// Recursive top-down build. Splits the [begin, end) range of leaves
// along the longest centroid-axis at the median, then writes the
// internal/leaf node IN PLACE at out_nodes[dst_idx] (caller pre-allocates
// the slot so the parent can guarantee its two children's indices are
// CONTIGUOUS: right = left + 1, the invariant the shader's traversal
// relies on in pickLightFromTree()).
//
// Slot allocation pattern at an internal node:
//   dst_idx               = parent (pre-allocated by caller)
//   children_first        = out_nodes.size() at entry; reserve TWO
//                           slots here -- left at children_first,
//                           right at children_first + 1.
//   Then recurse to fill each child's subtree.
//
// This replaces an earlier append-then-recurse scheme that was buggy:
// the previous code reserved only the parent slot then let the left
// recursive call append its own slot, which made the left ROOT live at
// (parent + 1) and the right ROOT live at (parent + 1 + left_subtree_size)
// -- so right = left + 1 was true ONLY when the left subtree was a
// single leaf (#177).  At N>=4 the right subtree became unreachable
// because the shader follows left_first+1, which on the original code
// pointed back into the LEFT subtree's interior.
//
// Stack depth is ceil(log2 N); N=1000 -> depth 10, well under any
// reentrant cap.  We pre-reserved 2N slots before the first recurse
// in the caller so writes via dst_idx never trigger a realloc; the
// children_first appends do trigger growth but never invalidate any
// dst_idx slot (we always write the parent slot AFTER the children
// have been built, but we hold dst_idx as an INDEX, not a pointer).
void BuildRecursive(std::vector<LightSummary>& leaves,
                    std::vector<LightTreeNode>& out_nodes,
                    std::uint32_t begin, std::uint32_t end,
                    std::uint32_t dst_idx) {
    const std::uint32_t n = end - begin;

    if (n == 1) {
        WriteLeafAt(leaves[begin], out_nodes, dst_idx);
        return;
    }

    // Compute centroid AABB to choose the split axis. Using centroid
    // AABB (not the union-extents AABB) gives a better split when one
    // light has a huge influence radius and would otherwise dominate
    // the axis pick.
    float c_min[3] = { std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity() };
    float c_max[3] = { -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity() };
    for (std::uint32_t i = begin; i < end; ++i) {
        const LightSummary& s = leaves[i];
        for (int k = 0; k < 3; ++k) {
            const float c = 0.5f * (s.aabb_min[k] + s.aabb_max[k]);
            c_min[k] = std::min(c_min[k], c);
            c_max[k] = std::max(c_max[k], c);
        }
    }
    int axis = 0;
    float best_ext = c_max[0] - c_min[0];
    for (int k = 1; k < 3; ++k) {
        const float e = c_max[k] - c_min[k];
        if (e > best_ext) { best_ext = e; axis = k; }
    }

    // Median split: partition by centroid on the chosen axis. nth_element
    // is O(n) average and gives a balanced tree. If all centroids are
    // exactly coincident (e.g. 200 lights spawned at the origin), fall
    // back to equal halves to avoid infinite recursion.
    std::uint32_t mid = begin + n / 2;
    if (best_ext < 1e-6f) {
        // Coincident: just halve the index range.
    } else {
        std::nth_element(leaves.begin() + begin,
                         leaves.begin() + mid,
                         leaves.begin() + end,
                         [axis](const LightSummary& a, const LightSummary& b) {
                             const float ca = 0.5f * (a.aabb_min[axis] + a.aabb_max[axis]);
                             const float cb = 0.5f * (b.aabb_min[axis] + b.aabb_max[axis]);
                             return ca < cb;
                         });
    }

    // Reserve two CONTIGUOUS child slots before recursing. The shader
    // computes right = left + 1, so both children must live at adjacent
    // indices. We append two placeholders here -- their indices are
    // guaranteed adjacent regardless of how many nodes each subtree
    // eventually produces, because each subtree's own recursion gets a
    // pre-allocated dst slot and only emits FURTHER child placeholders
    // (which sit beyond our right child's slot in the vector).
    const std::uint32_t left_idx  = static_cast<std::uint32_t>(out_nodes.size());
    out_nodes.emplace_back();   // child A placeholder
    out_nodes.emplace_back();   // child B placeholder
    const std::uint32_t right_idx = left_idx + 1u;

    BuildRecursive(leaves, out_nodes, begin, mid, left_idx);
    BuildRecursive(leaves, out_nodes, mid,   end, right_idx);

    // Combine child summaries from the just-written child nodes. The
    // child nodes hold the same cluster summary we just computed
    // bottom-up; pull them straight out instead of re-deriving from
    // the leaves vector (which has been partially rearranged by
    // nth_element above and no longer matches the [begin, mid) /
    // [mid, end) partition we just consumed -- the partition was
    // CORRECT at recurse time, but the bookkeeping is cleaner if we
    // just read what we wrote).
    //
    // IMPORTANT: each child's recursive subtree may have done many
    // emplace_back()s into out_nodes. The vector may have reallocated.
    // We never hold a pointer or reference across the recursive call,
    // only indices.
    LightTreeNode& self = out_nodes[dst_idx];
    const LightTreeNode& cl = out_nodes[left_idx];
    const LightTreeNode& cr = out_nodes[right_idx];
    UnionAabb(self.aabb_min, self.aabb_max,
              cl.aabb_min, cl.aabb_max,
              cr.aabb_min, cr.aabb_max);
    self.intensity  = cl.intensity + cr.intensity;
    CombineCones(cl.cone_axis, cl.cone_cos_half,
                  cr.cone_axis, cr.cone_cos_half,
                  self.cone_axis, self.cone_cos_half);
    self.left_first = left_idx;
    self.count      = 0u;  // internal
}

}  // namespace

void BuildLightTree(const std::vector<LightInput>& lights,
                    LightTree& dst) {
    dst.Clear();
    if (lights.empty()) return;

    // Per-light leaf summaries in upload order. After this point the
    // build sorts the summary vector in place -- the original `lights`
    // vector is NOT mutated. The `light_id` field on each summary keeps
    // the leaf's edge back to its slot in the upload buffer through the
    // partitioning.
    std::vector<LightSummary> leaves;
    leaves.reserve(lights.size());
    for (std::uint32_t i = 0; i < lights.size(); ++i) {
        leaves.push_back(MakeLeafSummary(lights[i], i));
    }

    // Reserve worst-case capacity up front. The 2N-1 bound is for a
    // full binary tree with N leaves. We allocate one root slot here
    // and let BuildRecursive append children pair-wise into the
    // remaining capacity; the reservation prevents grow-and-realloc
    // mid-build so any references we DO take (we only take indices,
    // but the reserved capacity also avoids cache-thrash on rebuild).
    dst.nodes.reserve(2u * lights.size());
    dst.nodes.emplace_back();   // root slot

    BuildRecursive(leaves, dst.nodes, 0u,
                    static_cast<std::uint32_t>(leaves.size()),
                    /*dst_idx=*/0u);

    dst.light_count = static_cast<std::uint32_t>(lights.size());
    dst.root_index  = 0u;
}

// --- AsyncLightTreeBuilder -------------------------------------------------
// Persistent worker thread. The work itself is bog-standard CPU code
// (BuildLightTree is pure host-side, no GPU API surface), so the worker
// is a free thread with no special scheduling. We don't use the engine's
// JobSystem here because:
//   - The build is bursty / dirty-bit-gated, not parallel-divisible,
//     so a one-shot std::thread carries no per-call setup hit.
//   - Avoiding the dep on src/core/Jobs/ keeps the renderer lib's
//     dependency boundary the same shape it was pre-PR (engine -> renderer
//     only; no engine internals leaking the other way).
AsyncLightTreeBuilder::AsyncLightTreeBuilder() {
    worker_ = std::thread([this] { this->WorkerLoop(); });
}

AsyncLightTreeBuilder::~AsyncLightTreeBuilder() {
    {
        std::lock_guard<std::mutex> lk(input_mu_);
        stop_ = true;
        // Drop any pending input so the worker doesn't try to do one
        // more build before checking `stop_`.
        has_pending_input_ = false;
        pending_inputs_.clear();
    }
    input_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void AsyncLightTreeBuilder::SubmitInputs(std::vector<LightInput>&& inputs) {
    {
        std::lock_guard<std::mutex> lk(input_mu_);
        // Replace any not-yet-consumed snapshot. If the worker is
        // already building from a previous snapshot, that build still
        // finishes -- its output just gets overwritten when this new
        // snapshot's build completes. That's intentional: we want the
        // freshest result, not a queue of stale ones.
        pending_inputs_   = std::move(inputs);
        has_pending_input_ = true;
        build_in_flight_.store(true, std::memory_order_release);
    }
    input_cv_.notify_one();
}

const LightTree* AsyncLightTreeBuilder::TryAcquireResult() {
    const std::uint32_t gen = result_generation_.load(std::memory_order_acquire);
    if (gen == last_consumed_generation_) return nullptr;
    // New tree available. Take the result mutex so we don't race with
    // the worker flipping `active_slot_` between our load + the read.
    std::lock_guard<std::mutex> lk(result_mu_);
    last_consumed_generation_ = result_generation_.load(std::memory_order_acquire);
    return &slots_[active_slot_];
}

void AsyncLightTreeBuilder::WorkerLoop() {
    std::vector<LightInput> local;
    while (true) {
        // 1. Wait for a job (or shutdown).
        {
            std::unique_lock<std::mutex> lk(input_mu_);
            input_cv_.wait(lk, [this] {
                return stop_.load() || has_pending_input_;
            });
            if (stop_.load()) return;
            local = std::move(pending_inputs_);
            pending_inputs_.clear();
            has_pending_input_ = false;
        }

        // 2. Build into the INACTIVE slot. The active slot is being
        //    read by main; we don't hold result_mu_ for the build
        //    itself (the expensive step) so main can keep consuming
        //    the prior tree while we work.
        int write_slot;
        {
            std::lock_guard<std::mutex> lk(result_mu_);
            write_slot = 1 - active_slot_;
        }
        BuildLightTree(local, slots_[write_slot]);

        // 3. Publish the new slot. Single atomic store advances the
        //    generation counter; the next TryAcquireResult will see
        //    the new gen and read from the slot we just wrote.
        {
            std::lock_guard<std::mutex> lk(result_mu_);
            active_slot_ = write_slot;
        }
        result_generation_.fetch_add(1u, std::memory_order_acq_rel);
        build_in_flight_.store(false, std::memory_order_release);
        // Loop back to wait for the next job. If main already kicked
        // another one while we were building, has_pending_input_ is
        // true and the condvar wait above returns immediately.
    }
}
// --- end AsyncLightTreeBuilder ---------------------------------------------

}  // namespace pt::renderer

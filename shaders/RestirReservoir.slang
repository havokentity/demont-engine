// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// ReSTIR DI Phase A (issue #78) -- Shared Reservoir GPU module.
//
// This Slang module owns the GPU-side `Reservoir` struct + its
// pack/unpack helpers + the WRS update primitives. PathTrace.slang
// (initial candidate generation), RestirTemporal.slang (temporal
// reuse), RestirSpatial.slang (separable spatial reuse), and
// RestirFinal.slang (shadow test + Lambert shading) all import it so
// the encoding stays a single source of truth.
//
// HOST MIRROR: src/renderer/RestirReservoir.h. Layout must stay
// byte-identical (the host has a static_assert sizeof(Reservoir) == 64).
//
// Encoding: 4 × float4 lanes = 64 B per pixel. Slang's
// StructuredBuffer<Reservoir> would emit padding on MSL because the
// struct has mixed uint/float lanes; we encode as
// StructuredBuffer<float4> and asuint() the integer lanes manually.
// Same pattern PathTrace.slang's loadLight uses (the analytic light
// list is also `float4` strided).
//
// Algorithm reference: Bitterli et al. "Spatiotemporal Reservoir
// Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting,"
// SIGGRAPH 2020. Pseudocode for the WRS / temporal / spatial primitives
// is the paper's Algorithm 2 / Algorithm 4; comments below cite line
// numbers from that paper where helpful.

// --- ReSTIR Phase A constants (mirror host RestirReservoir.h) -------------
static const uint kRestirMaxK            = 64u;
static const uint kRestirTemporalMaxM    = 20u;
static const uint kRestirSpatialNbrs     = 4u;
static const float kRestirSpatialRadiusPx = 10.0f;

static const uint kRestirBiasBiased   = 0u;
static const uint kRestirBiasUnbiased = 1u;

// Per-pixel reservoir stride in float4 lanes. 4 lanes = 64 B. Matches
// the host `Reservoir` struct alignment.
static const uint kReservoirStrideFloat4 = 4u;

static const uint kRestirInvalidLight = 0xFFFFFFFFu;

// --- Reservoir struct -----------------------------------------------------
// Mirror of host src/renderer/RestirReservoir.h::Reservoir. Field order
// MUST match (and the host's static_assert catches any drift).
struct Reservoir {
    // Lane 0
    uint  light_idx;     // index into light_prims; kRestirInvalidLight = empty
    float p_hat;         // target pdf at the receiver (unshadowed Lambert estimate)
    float w_sum;         // running sum of source/target ratios (WRS state)
    uint  M;             // # candidates merged so far; capped by kTemporalMaxM

    // Lane 1: sample point on the light surface (world space)
    float3 light_pos;
    float  W;            // final resampling weight = w_sum / (M * p_hat)

    // Lane 2: light surface normal; (0,1,0) for point/spot delta lights
    float3 light_n;
    float  pdf_solid;    // original solid-angle sampling pdf (Veach-correct)

    // Lane 3: cached emission radiance + shadow-ray distance
    float3 Le;
    float  dist;
};

// Construct an empty reservoir. The WRS update primitives below treat
// p_hat=0 / M=0 as "no candidate yet", so callers can streamUpdate
// without any extra branches.
Reservoir initReservoir() {
    Reservoir r;
    r.light_idx = kRestirInvalidLight;
    r.p_hat     = 0.0;
    r.w_sum     = 0.0;
    r.M         = 0u;
    r.light_pos = float3(0.0);
    r.W         = 0.0;
    r.light_n   = float3(0.0, 1.0, 0.0);
    r.pdf_solid = 0.0;
    r.Le        = float3(0.0);
    r.dist      = 0.0;
    return r;
}

// Encode 4 float4 lanes from a Reservoir. uint -> float via asfloat()
// preserves the bit pattern; the loader recovers it with asuint().
void packReservoir(Reservoir r, out float4 v0, out float4 v1,
                   out float4 v2, out float4 v3) {
    v0 = float4(asfloat(r.light_idx), r.p_hat, r.w_sum, asfloat(r.M));
    v1 = float4(r.light_pos, r.W);
    v2 = float4(r.light_n,   r.pdf_solid);
    v3 = float4(r.Le,        r.dist);
}

Reservoir unpackReservoir(float4 v0, float4 v1, float4 v2, float4 v3) {
    Reservoir r;
    r.light_idx = asuint(v0.x);
    r.p_hat     = v0.y;
    r.w_sum     = v0.z;
    r.M         = asuint(v0.w);
    r.light_pos = v1.xyz;
    r.W         = v1.w;
    r.light_n   = v2.xyz;
    r.pdf_solid = v2.w;
    r.Le        = v3.xyz;
    r.dist      = v3.w;
    return r;
}

// Load a reservoir from a StructuredBuffer<float4> at the given pixel
// linear index. Used by every Restir kernel. The buffer is laid out
// y-major (pixel_idx = y * width + x), 4 float4s per pixel.
Reservoir loadReservoir(StructuredBuffer<float4> buf, uint pixel_idx) {
    uint base = pixel_idx * kReservoirStrideFloat4;
    return unpackReservoir(buf[base + 0u], buf[base + 1u],
                           buf[base + 2u], buf[base + 3u]);
}

// Store a reservoir into an RWStructuredBuffer<float4>. Pair-symmetric
// with loadReservoir.
void storeReservoir(RWStructuredBuffer<float4> buf, uint pixel_idx, Reservoir r) {
    uint base = pixel_idx * kReservoirStrideFloat4;
    float4 v0, v1, v2, v3;
    packReservoir(r, v0, v1, v2, v3);
    buf[base + 0u] = v0;
    buf[base + 1u] = v1;
    buf[base + 2u] = v2;
    buf[base + 3u] = v3;
}

// --- WRS streaming primitives ---------------------------------------------
// Bitterli §3.1 Algorithm 2. `w_i = source_pdf_i / target_pdf_i` is
// the WRS weight; the reservoir picks one survivor with probability
// proportional to w_i.

// Streaming update for a single candidate sample. Returns true when
// the candidate became the new survivor (caller can save the cached
// emission / pdfs into the reservoir struct).
//
//   r          : reservoir (in-out)
//   p_hat_x    : target pdf evaluated at the candidate (un-shadowed
//                Lambert estimate length(Le * brdf * cos) / dist^2)
//   source_pdf : sampling pdf the candidate was drawn from (pmf *
//                solid-angle pdf for area lights, pmf alone for delta
//                lights)
//   u          : uniform [0,1) variate for the survivor draw
bool streamingUpdate(inout Reservoir r, float p_hat_x, float source_pdf, float u) {
    // RIS weight: target/source, clamped to keep degenerate samples
    // (source_pdf -> 0) from blowing the running sum.
    float w_i = (source_pdf > 1e-9) ? (p_hat_x / source_pdf) : 0.0;
    r.w_sum += w_i;
    r.M     += 1u;
    if (r.w_sum > 0.0 && u < (w_i / r.w_sum)) {
        // Caller fills the survivor's sample fields after this returns
        // true; we only update the bookkeeping here.
        r.p_hat = p_hat_x;
        return true;
    }
    return false;
}

// Merge a foreign reservoir (temporal or spatial neighbour) into `r`.
// Treats the neighbour's survivor as ONE candidate with combined weight
// w = neighbour.W * p_hat_of_neighbour_at_receiver * neighbour.M.
// This is the standard combineReservoirs from Bitterli Algorithm 4.
//
// p_hat_other_at_receiver lets the caller re-evaluate the neighbour's
// target pdf at THIS pixel's surface (which is what makes spatial reuse
// work: a light that's bright at the neighbour might be back-facing
// here, in which case p_hat=0 and the merge has zero weight).
bool combineReservoir(inout Reservoir r,
                      Reservoir other,
                      float p_hat_other_at_receiver,
                      float u) {
    // Inactive neighbour (no candidate, sample count 0) is a no-op.
    if (other.M == 0u || other.p_hat <= 0.0) {
        return false;
    }
    // RIS weight for the neighbour. Use the receiver-side target pdf
    // so a chosen light that's back-facing here weighs zero. W *
    // p_hat_other_at_receiver * M is the standard "merge a reservoir
    // as one sample" weight from Bitterli §4.
    float w_i = p_hat_other_at_receiver * other.W * float(other.M);
    r.w_sum += w_i;
    r.M     += other.M;
    if (r.w_sum > 0.0 && u < (w_i / r.w_sum)) {
        // Adopt the foreign survivor's sample. p_hat updates to the
        // receiver-side target pdf so the survivor's bookkeeping
        // remains internally consistent at this pixel.
        r.p_hat     = p_hat_other_at_receiver;
        r.light_idx = other.light_idx;
        r.light_pos = other.light_pos;
        r.light_n   = other.light_n;
        r.pdf_solid = other.pdf_solid;
        r.Le        = other.Le;
        r.dist      = other.dist;
        return true;
    }
    return false;
}

// Finalize: compute the per-sample resampling weight W. Called after
// all streamingUpdate / combineReservoir merges. The estimator then
// evaluates F = f(survivor) * W where f is the true integrand at the
// receiver (Lambert + shadow visibility); the W absorbs the RIS
// correction.
void finalizeReservoir(inout Reservoir r) {
    if (r.M == 0u || r.p_hat <= 1e-9) {
        r.W = 0.0;
        return;
    }
    r.W = r.w_sum / (float(r.M) * r.p_hat);
}

// Compute the target pdf p_hat(x) for a candidate at the given
// receiver point. The un-shadowed Lambert estimator is
//   p_hat = length(Le * brdf * cos(n, l)) / dist^2   (point/spot)
//   p_hat = length(Le * brdf * cos(n, l) * cos_light) / dist^2 / area_pdf
// We use the luminance of the radiance contribution as the scalar
// estimate (Veach's standard choice for spectral-collapsed RIS).
//
// For Phase A we pre-bake the (1/dist^2) and cos_light into the
// caller's Le, mirroring PathTrace.slang's sampleAnalyticLight return.
// This helper then just dots in the receiver's cos(n, l) and returns
// luminance. brdf is folded later in the shading stage so this stays a
// dimensionally-correct scalar pdf.
float reservoirTargetPdf(float3 hit_n, float3 to_light_dir, float3 Le_pre) {
    float n_dot_l = max(dot(hit_n, to_light_dir), 0.0);
    if (n_dot_l <= 0.0) return 0.0;
    float3 contrib = Le_pre * n_dot_l;
    // Rec.709 luminance — same weights PathTrace.slang's SIGMA-style
    // shadow demod uses, for cross-pass tonal consistency.
    return max(dot(contrib, float3(0.2126, 0.7152, 0.0722)), 0.0);
}

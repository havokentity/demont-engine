// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Planetary P5 (#259) / water roadmap #133 Phase 2: the ocean cascade
// solver as a compute pre-pass.
//
// WHAT THIS REPLACES, AND WHAT IT MEASURED
// ----------------------------------------
// pt::ocean::OceanFFT is a single-threaded CPU Tessendorf solver. With the
// three planetary cascades it runs FIFTEEN inverse 2-D FFTs per frame
// (5 spectra x 3 cascades), packs three grid^2 RGBA32F fields, folds three
// foam passes over them, concatenates the result into a stacked atlas and
// re-uploads 6 MB. Measured on an M4 Max, 120 frames of
// tests/goldens/scenes/planet_ocean_boat.cfg at r_ocean_grid_size 256:
// 6.39 s with the cascades, 4.52 s with r_ocean 0 -- 15.6 ms per frame on
// one thread, i.e. a 64 fps ceiling before the path tracer does anything.
//
// THE HOST STAYS THE REFERENCE. The static base spectrum H0(k) and its
// conjugate mirror conj(H0(-k)) are still built on the CPU, because they
// are drawn from a seeded std::mt19937 and "the same Gaussians" is not a
// thing a GPU can promise. They are uploaded ONCE PER SPECTRUM REBUILD
// (a wind / grid / tile / amplitude / seed / band change), not per frame.
// What moves here is the per-frame work: the time evolution, the five
// spectra, the fifteen inverse FFTs, the pack, the normals and the foam.
// pt::ocean::OceanFFT keeps running under tests/pt_ocean_fft_test.cpp and
// stays the fallback on every backend without this pipeline.
//
// DETERMINISM IS NOT NEGOTIABLE. The three P5 golden cells render
// byte-identical run to run in Release AND Debug, and a GPU solver that
// gave that up would be worse than the CPU one however fast it was. So:
//
//   * every reduction here is over max(), which is exact and both
//     associative and commutative in IEEE-754 -- reordering it cannot
//     change a bit -- and it is still done in a FIXED tree order in
//     groupshared memory rather than through an atomic, so the order is
//     not merely irrelevant, it is pinned;
//   * there is no floating-point SUM reduction anywhere in this file;
//   * every butterfly reads and writes indices that belong to exactly one
//     thread in its stage, separated by GroupMemoryBarrierWithGroupSync;
//   * nothing is read back to the host, so no frame's result can depend on
//     when a previous frame's copy landed.
//
// WHAT IS NOT BIT-IDENTICAL TO THE CPU, AND WHY THAT IS FINE. The twiddle
// factors. OceanFFT::Ifft1D advances w by repeated complex multiplication
// (`w *= wlen`), which drifts by O(k * u) over a stage of length 2k; this
// kernel evaluates cos/sin at the exact angle instead. That is the single
// largest term in the CPU-vs-GPU difference and it is the CPU's error, not
// this one's. See tests/pt_ocean_gpu_test.cpp, which derives the bound and
// measures against it.
//
// PASS STRUCTURE (one entry point, dispatched five times with ctl.x as the
// pass index -- the same shape GodRays.slang uses for its two passes):
//
//   0  ROWS     one threadgroup per (row, field, cascade). Evaluates
//               H(k,t) and the four derived spectra inline while loading,
//               so the spectrum never round-trips through memory, runs the
//               radix-2 inverse FFT in groupshared, writes to scratch.
//   1  COLS     same, over columns of the scratch, in place.
//   2  PACK     one thread per texel: the (-1)^(x+y) centre-to-corner sign
//               fold, the displacement + normal write, and each
//               threadgroup's peak |height| into the reduce buffer.
//   3  REDUCE   one threadgroup: per-cascade peak |height| from the
//               per-threadgroup maxima, and the march bracket into the
//               atlas metadata row.
//   4  FOAM     one thread per texel: the Jacobian-fold foam, the
//               whitecap-coverage ramp and the persistence trail, written
//               into ocean_displacement.w.
//
// The host side of all of this -- the pass indices, the push layout and
// the dispatch shapes -- lives in ONE place,
// src/physics/OceanCascadeDispatch.h, shared verbatim by
// Engine::DispatchOceanCascades and tests/pt_ocean_gpu_test.cpp. A wire
// format with two transcriptions is a wire format that drifts.

// --- Resources ------------------------------------------------------------
//
// vk::binding numbers start at 40, past every binding the shared PathTrace
// descriptor-set layout declares today (the highest is 39). They are
// DECLARED so the SPIR-V is well-formed and spirv-val passes, but no
// Vulkan pipeline is built from this module yet -- VulkanDevice's shared
// layout would need all six entries added first (a shader declaring a
// binding the layout lacks is a hard vkCreateComputePipelines failure even
// under PARTIALLY_BOUND), and no Vulkan device has executed any of this
// planetary arc (#290). Until then Vulkan takes the CPU cascade solver.
//
// ocean_h0 packs BOTH halves of the static spectrum per cell:
//   .xy = H0(k), .zw = conj(H0(-k)).
// Uploaded by the host on a spectrum rebuild. Keeping the mirror on the
// host rather than recomputing it here means the GPU never has to know the
// [-N/2, N/2) lattice's mirror rule, and the two cannot drift apart.
[[vk::binding(40, 0)]] RWStructuredBuffer<float4> ocean_h0;
// Row-major complex scratch, [cascade][field][y][x]. 5 fields x 3 cascades
// x grid^2 x 8 B = 7.5 MiB at grid 256, 30.0 MiB at 512.
[[vk::binding(41, 0)]] RWStructuredBuffer<float2> ocean_scratch;
// Persistent foam accumulator, [cascade][y][x]. The one piece of state
// that survives a frame: whitecaps linger after a crest breaks.
[[vk::binding(42, 0)]] RWStructuredBuffer<float>  ocean_foam_accum;
// [cascade][threadgroup] peak |height| from PASS_PACK, then
// [cascades * tiles + c] the per-cascade peak from PASS_REDUCE.
[[vk::binding(43, 0)]] RWStructuredBuffer<float>  ocean_reduce;
[[vk::binding(44, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> ocean_disp;
[[vk::binding(45, 0), vk::image_format("rgba32f")]] RWTexture2D<float4> ocean_nrm;

// The push block is 80 B, comfortably under VulkanDevice::kPushSplitOffset
// (112 B), so unlike PathTrace / CloudsRaymarch it needs no Frame-UBO tail
// and the same declaration serves Metal and SPIR-V.
[[vk::push_constant]]
cbuffer Push {
    // .x pass index, .y grid size N, .z cascade count, .w reserved.
    uint4  ctl;
    // .x absolute sim time t (s), .y choppiness lambda,
    // .z foam Jacobian threshold, .w foam amount.
    float4 p0;
    // .x foam persistence decay for this frame's dt (host-side pow, so no
    //    transcendental disagreement with the CPU reference lives here),
    // .y wind whitecap coverage in [0,1] (host-side smoothstep, same
    //    reason), .z march-bracket scale, .w march-bracket bias (m).
    float4 p1;
    // Cascade tile periods in metres. .w unused.
    float4 period;
    // .x gravity (m/s^2). Feeds omega = sqrt(g |k|), recomputed here
    // rather than uploaded: it is a pure function of the same kx / kz this
    // kernel already forms, in the same order, so it is bit-identical to
    // OceanFFT::RebuildBaseSpectrum's omega_ and costs one buffer less.
    float4 p2;
};

static const float kTwoPi  = 6.28318530717958647692;
// Largest grid the engine snaps r_ocean_grid_size to (see StepOcean).
// 512 complex singles = 4 KB of threadgroup memory, well inside Metal's
// 32 KB budget.
static const uint  kMaxGrid = 512u;
static const uint  kFields  = 5u;   // h, dh/dx, dh/dz, Dx, Dz
// One threadgroup is [numthreads(8,8,1)] because MetalCommandBuffer::
// Dispatch hardcodes that shape for every pipeline in the engine. The FFT
// lanes therefore stride by 64, not by the grid size.
static const uint  kLanes   = 64u;

static const uint PASS_ROWS   = 0u;
static const uint PASS_COLS   = 1u;
static const uint PASS_PACK   = 2u;
static const uint PASS_REDUCE = 3u;
static const uint PASS_FOAM   = 4u;

groupshared float2 g_line[kMaxGrid];
groupshared float  g_tile[kLanes];
groupshared float  g_cmax[4];

// Complex multiply in the same operand order std::complex<float>'s
// operator* uses, so the only difference from the host is the rounding of
// the same three-term expressions.
float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint log2u(uint n) {
    uint l = 0u;
    while ((1u << l) < n) ++l;
    return l;
}

uint bitrev(uint x, uint bits) {
    uint r = 0u;
    for (uint i = 0u; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

uint specIndex(uint N, uint c, uint f, uint x, uint y) {
    return ((c * kFields + f) * N + y) * N + x;
}
uint cellIndex(uint N, uint c, uint x, uint y) {
    return (c * N + y) * N + x;
}

// H(k,t) and the four spectra derived from it, for one lattice cell.
//
// Mirrors OceanFFT::Update's inner loop term for term:
//   k     = 2 pi (n, m) / L,  n = ix - N/2,  m = iy - N/2
//   omega = sqrt(g |k|)
//   H     = H0 exp(+i omega t) + conj(H0(-k)) exp(-i omega t)
//   slope = (i kx H, i kz H)          -- gradient of the height field
//   chop  = (-i kx/|k| H, -i kz/|k| H) -- Gerstner lateral displacement
float2 spectrumAt(uint N, uint c, uint f, uint ix, uint iy,
                  float L, float t, float g) {
    int half_ = int(N) / 2;
    int n = int(ix) - half_;
    int m = int(iy) - half_;
    float kx = kTwoPi * float(n) / L;
    float kz = kTwoPi * float(m) / L;
    float k2 = kx * kx + kz * kz;
    float k  = sqrt(k2);

    float wt = sqrt(g * k) * t;
    float cw = cos(wt);
    float sw = sin(wt);

    float4 s = ocean_h0[cellIndex(N, c, ix, iy)];
    float2 h = cmul(s.xy, float2(cw, sw)) + cmul(s.zw, float2(cw, -sw));

    if (f == 0u) return h;
    if (f == 1u) return cmul(float2(0.0, kx), h);
    if (f == 2u) return cmul(float2(0.0, kz), h);
    // The DC cell has no direction, and OceanFFT leaves its choppy spectra
    // at zero rather than dividing by |k| = 0.
    if (k > 1e-6) {
        float inv_k = 1.0 / k;
        if (f == 3u) return cmul(float2(0.0, -kx * inv_k), h);
        return cmul(float2(0.0, -kz * inv_k), h);
    }
    return float2(0.0, 0.0);
}

// In-place radix-2 inverse FFT of g_line[0 .. N), assuming the caller
// already deposited the input in bit-reversed order.
//
// +i exponent (inverse) and NO 1/N normalisation, matching
// OceanFFT::Ifft1D / Ifft2D: Tessendorf's synthesis evaluates
// h(x) = sum_k h~(k) exp(i k.x), the un-normalised inverse DFT, because
// the spectral amplitudes already carry the physical per-mode height.
//
// Each stage's butterfly b owns indices i0 and i1 and no other lane
// touches them, so the only synchronisation needed is the stage barrier.
void ifft1d(uint N, uint lane) {
    for (uint len = 2u; len <= N; len <<= 1) {
        uint half_ = len >> 1;
        uint total = N >> 1;
        for (uint b = lane; b < total; b += kLanes) {
            uint block = b / half_;
            uint k     = b - block * half_;
            uint i0    = block * len + k;
            uint i1    = i0 + half_;
            float ang  = kTwoPi * float(k) / float(len);
            float2 w   = float2(cos(ang), sin(ang));
            float2 a   = g_line[i0];
            float2 tv  = cmul(w, g_line[i1]);
            g_line[i0] = a + tv;
            g_line[i1] = a - tv;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID,
          uint3 gid  : SV_GroupID,
          uint3 gtid : SV_GroupThreadID) {
    uint pass  = ctl.x;
    uint N     = max(ctl.y, 1u);
    uint count = max(ctl.z, 1u);
    uint lane  = gtid.y * 8u + gtid.x;
    float t    = p0.x;
    float g    = p2.x;

    if (pass == PASS_ROWS || pass == PASS_COLS) {
        // gid = (line index, field, cascade). One threadgroup per line.
        uint line = gid.x;
        uint f    = gid.y;
        uint c    = gid.z;
        if (line >= N || f >= kFields || c >= count) return;
        uint bits = log2u(N);
        float L   = (c == 0u) ? period.x : ((c == 1u) ? period.y : period.z);

        if (pass == PASS_ROWS) {
            // Row `line`, x running. The spectrum is evaluated here rather
            // than in a pass of its own: it saves a full grid^2 x 5 x 3
            // write plus read, and cos/sin of the same wt five times is
            // cheaper than that round trip.
            for (uint i = lane; i < N; i += kLanes) {
                g_line[bitrev(i, bits)] =
                    spectrumAt(N, c, f, i, line, L, t, g);
            }
        } else {
            // Column `line`, y running.
            for (uint i = lane; i < N; i += kLanes) {
                g_line[bitrev(i, bits)] =
                    ocean_scratch[specIndex(N, c, f, line, i)];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        ifft1d(N, lane);
        if (pass == PASS_ROWS) {
            for (uint i = lane; i < N; i += kLanes) {
                ocean_scratch[specIndex(N, c, f, i, line)] = g_line[i];
            }
        } else {
            for (uint i = lane; i < N; i += kLanes) {
                ocean_scratch[specIndex(N, c, f, line, i)] = g_line[i];
            }
        }
        return;
    }

    if (pass == PASS_PACK) {
        uint ix = dtid.x, iy = dtid.y, c = dtid.z;
        float peak = 0.0;
        if (ix < N && iy < N && c < count) {
            // (-1)^(x+y) folds the inverse FFT's centre-origin output into
            // a corner-origin tile that wraps seamlessly under the
            // world -> uv lookup.
            float sign   = ((ix + iy) & 1u) != 0u ? -1.0 : 1.0;
            float lambda = p0.y;

            float h  = ocean_scratch[specIndex(N, c, 0u, ix, iy)].x * sign;
            float gx = ocean_scratch[specIndex(N, c, 1u, ix, iy)].x * sign;
            float gz = ocean_scratch[specIndex(N, c, 2u, ix, iy)].x * sign;
            float dX = ocean_scratch[specIndex(N, c, 3u, ix, iy)].x * sign * lambda;
            float dZ = ocean_scratch[specIndex(N, c, 4u, ix, iy)].x * sign * lambda;

            uint2 texel = uint2(ix, c * N + iy);
            // .w (foam) is written by PASS_FOAM, which needs this pass's
            // neighbours; zero here so a foam pass that never ran leaves
            // no foam rather than stale foam.
            ocean_disp[texel] = float4(dX, h, dZ, 0.0);

            // n = normalize(-dh/dx, 1, -dh/dz). Written as v * (1/sqrt(d))
            // rather than v / sqrt(d) because that is what glm::normalize
            // does (glm::inversesqrt(float) is 1.0f / std::sqrt(x)), and
            // one rounding of agreement is free.
            float3 v = float3(-gx, 1.0, -gz);
            float3 nr = v * (1.0 / sqrt(dot(v, v)));
            ocean_nrm[texel] = float4(nr, 1.0);

            peak = abs(h);
        }
        // Per-threadgroup peak |height|, fixed-order tree. max() is exact
        // in IEEE-754, so this is bit-identical whatever order the lanes
        // arrive in -- and the order is pinned anyway.
        g_tile[lane] = peak;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = kLanes >> 1; s > 0u; s >>= 1) {
            if (lane < s) g_tile[lane] = max(g_tile[lane], g_tile[lane + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u) {
            uint tiles_x = (N + 7u) / 8u;
            uint tiles   = tiles_x * tiles_x;
            ocean_reduce[gid.z * tiles + gid.y * tiles_x + gid.x] = g_tile[0];
        }
        return;
    }

    if (pass == PASS_REDUCE) {
        uint tiles_x = (N + 7u) / 8u;
        uint tiles   = tiles_x * tiles_x;
        if (lane < count) {
            float m = 0.0;
            for (uint i = 0u; i < tiles; ++i) {
                m = max(m, ocean_reduce[lane * tiles + i]);
            }
            g_cmax[lane] = m;
        } else if (lane < 4u) {
            g_cmax[lane] = 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            for (uint c = 0u; c < count; ++c) {
                ocean_reduce[count * tiles + c] = g_cmax[c];
            }
        }
        // THE MARCH BRACKET, AND WHY IT LIVES IN THE TEXTURE.
        //
        // oceanRayMarchShell brackets the band at R_sea +/- h_max, where
        // h_max is the SUM of the cascades' peak crests (they are
        // independent fields and their peaks can coincide). On the CPU
        // path the host knew that number before it built the push block.
        // Here the peak only exists after this dispatch, and reading it
        // back would cost a stall and put the bracket a frame behind the
        // field it brackets. So it travels in the atlas instead: one
        // metadata row past the last cascade, which oceanBilinearTexelsC
        // can never address because its wrap is modulo `grid` inside the
        // cascade's own band. The host writes the identical value into the
        // identical texel on the CPU fallback path, so the two paths feed
        // the shader one number from one place.
        float total = g_cmax[0];
        if (count > 1u) total = total + g_cmax[1];
        if (count > 2u) total = total + g_cmax[2];
        float bracket = total * p1.z + p1.w;
        float4 meta = float4(bracket, g_cmax[0], g_cmax[1], g_cmax[2]);
        // Fill the whole row, not just texel 0: an unwritten RGBA32F texel
        // is whatever the allocation left there, and a metadata row that is
        // uniform cannot surprise a future reader that interpolates.
        for (uint i = lane; i < N; i += kLanes) {
            ocean_disp[uint2(i, count * N)] = meta;
        }
        return;
    }

    // PASS_FOAM.
    uint ix = dtid.x, iy = dtid.y, c = dtid.z;
    if (ix >= N || iy >= N || c >= count) return;

    float L    = (c == 0u) ? period.x : ((c == 1u) ? period.y : period.z);
    float dxz  = L / float(N);
    uint  base = c * N;
    uint2 self = uint2(ix, base + iy);
    float4 d   = ocean_disp[self];

    float lambda = p0.y;
    float thresh = p0.z;

    // Jacobian-fold foam. Where the choppy lateral displacement folds the
    // surface (det J < threshold) the crest is pinching. A round-swell sea
    // (lambda == 0) never folds, so its foam is the wind term alone.
    float foam_inst = 0.0;
    if (lambda > 0.0) {
        uint xp = (ix + 1u) % N;
        uint xm = (ix + N - 1u) % N;
        uint yp = (iy + 1u) % N;
        uint ym = (iy + N - 1u) % N;
        float4 rx = ocean_disp[uint2(xp, base + iy)];
        float4 lx = ocean_disp[uint2(xm, base + iy)];
        float4 uz = ocean_disp[uint2(ix, base + yp)];
        float4 dz = ocean_disp[uint2(ix, base + ym)];

        float dDxdx = (rx.x - lx.x) / (2.0 * dxz);
        float dDxdz = (uz.x - dz.x) / (2.0 * dxz);
        float dDzdx = (rx.z - lx.z) / (2.0 * dxz);
        float dDzdz = (uz.z - dz.z) / (2.0 * dxz);

        float jxx = 1.0 + dDxdx;
        float jzz = 1.0 + dDzdz;
        float det = jxx * jzz - dDxdz * dDzdx;

        foam_inst = clamp((thresh - det) / max(thresh, 1e-3), 0.0, 1.0);
    }

    // Wind-driven whitecap coverage, gated by crest height. p1.y is the
    // host's smoothstep over the Monahan/Wu whitecap-onset ramp; the crest
    // gate is the top 60% of THIS cascade's own peak, which is why the
    // reduce pass keeps the per-cascade maxima separately.
    uint  tiles_x = (N + 7u) / 8u;
    uint  tiles   = tiles_x * tiles_x;
    float cmax    = ocean_reduce[count * tiles + c];
    float h_norm  = (cmax > 1e-4) ? (1.0 / cmax) : 0.0;
    float wind_cov = p1.y;
    float wind_foam = 0.0;
    if (wind_cov > 0.0 && h_norm > 0.0) {
        float hn = d.y * h_norm;
        float crest = clamp((hn - 0.4) / 0.6, 0.0, 1.0);
        crest = crest * crest * (3.0 - 2.0 * crest);
        wind_foam = wind_cov * crest;
    }

    // Both terms are "fraction of surface that is froth", so a max (not a
    // sum) keeps coverage in [0,1].
    float foam_now = clamp(max(foam_inst, wind_foam) * p0.w, 0.0, 1.0);

    uint  fidx = cellIndex(N, c, ix, iy);
    float acc  = ocean_foam_accum[fidx] * p1.x;
    acc = max(acc, foam_now);
    acc = clamp(acc, 0.0, 1.0);
    ocean_foam_accum[fidx] = acc;

    d.w = acc;
    ocean_disp[self] = d;
}

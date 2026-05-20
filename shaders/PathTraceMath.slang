// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Pure-math helpers used by the path tracer.  Compiled to a Slang IR
// module (.slang-module) so PathTrace.slang's `import` resolves to a
// pre-compiled blob and ninja can build this module in parallel with
// the entry-point shader, plus skip the front-end re-parse on every
// PathTrace.spv rebuild.  Nothing in here references resource
// bindings or push constants -- it's deliberately self-contained so
// the same .slang-module file links into both the SPIR-V and Metal
// targets unmodified.

// PCG hash + uniform float in [0, 1).  Tiny state (one uint per pixel
// thread) -- the caller mutates the seed in place so successive draws
// from the same thread are decorrelated.
public uint pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}
public float randf(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed & 0x00FFFFFFu) / 16777216.0;
}

// Cosine-weighted hemisphere sample around `n`.  Used by the diffuse
// ambient term in the Lambert HDRI block (cosine weighting matches
// the BRDF, so the pdf cancels out and the estimator collapses to
// just sampling the sky in `amb_dir`).
public float3 sampleCosineHemisphere(float3 n, inout uint seed) {
    float u1 = randf(seed);
    float u2 = randf(seed);
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1));
    float3 ax = abs(n.x) > 0.9 ? float3(0,1,0) : float3(1,0,0);
    float3 t  = normalize(cross(n, ax));
    float3 b  = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

// The sun is a disc, not a point. Half-angular-radius from Earth is
// 0.00465 rad (~0.266 deg). Shadow rays sample a random direction
// inside that disc instead of the centre direction; over the temporal
// accumulator this gives soft penumbrae naturally (clouds, foliage,
// hard objects all gain real-physics-soft edges).
//
// This is *not* a softening hack -- the sun really is a disc and a
// point sample is the wrong physics. We're just doing the integral
// the path tracer was missing.
public static const float kSunAngularRadius = 0.00465;

public float3 sampleSunDisc(float3 sun_dir, inout uint seed) {
    float u1  = randf(seed);
    float u2  = randf(seed);
    // Uniform disc sample, scaled by sun's angular radius (small angle
    // so this approximates a uniform solid-angle sample on the disc).
    float r   = sqrt(u1) * kSunAngularRadius;
    float phi = 6.28318530718 * u2;
    float2 d  = float2(cos(phi), sin(phi)) * r;
    // Build an orthonormal basis around sun_dir and offset.
    float3 ax = abs(sun_dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t  = normalize(cross(ax, sun_dir));
    float3 b  = cross(sun_dir, t);
    return normalize(sun_dir + t * d.x + b * d.y);
}

// Schlick approximation of Fresnel reflectance.  r0 is the base
// reflectance at normal incidence; for dielectrics this is
// ((n1-n2)/(n1+n2))^2.
public float fresnelSchlick(float cos_theta, float r0) {
    float m = 1.0 - cos_theta;
    return r0 + (1.0 - r0) * m * m * m * m * m;
}

// --- Wave 8 spectral (#27) -------------------------------------------------
// Spectral dispersion helpers. Self-contained (no resource bindings / push
// refs) so they link into both the SPIR-V and Metal targets via the
// PathTraceMath IR module. The path tracer's hero-wavelength dielectric
// path (gated on r_spectral) uses these to bend each wavelength by its own
// real IOR and reconstruct RGB from the CIE 1931 colour-matching functions.

// Visible band the hero-wavelength sampler stratifies over, in nanometres.
public static const float kSpectralLambdaMin = 380.0;
public static const float kSpectralLambdaMax = 730.0;

// Stratified hero-wavelength sample for path sample index `s` of `n` total.
// Returns lambda in nanometres in [380, 730]. Stratification across the
// pixel's spp samples gives fast spectral convergence: instead of n random
// draws clustering, sample s lands in the s-th of n equal sub-bands and is
// jittered within it. With n == 1 this is a single uniform draw over the
// whole band (interactive 1-spp mode still disperses, it just needs the
// temporal accumulator to average many frames' worth of wavelengths).
public float heroWavelength(uint s, uint n, inout uint seed) {
    float ns = (n == 0u) ? 1.0 : float(n);
    float u  = (float(s) + randf(seed)) / ns;        // stratified [0,1)
    return kSpectralLambdaMin + u * (kSpectralLambdaMax - kSpectralLambdaMin);
}

// Cauchy dispersion model: n(lambda) = B + C / lambda^2, lambda in micrometres.
// `base_ior` is the material's nominal index (the RGB-path value, taken to be
// the index at the d-line, 587.6 nm). We solve for B so the curve passes
// through (base_ior, 587.6 nm) for the supplied dispersion coefficient C
// (units micrometre^2). Default crown-glass BK7 is C ~= 0.00420 um^2, giving
// the textbook ~0.008 spread of n across the visible band -- enough to split
// a beam into a clean rainbow without exaggeration. C == 0 collapses to the
// achromatic base_ior (used as the off-switch fallback so a non-dispersive
// material is bit-stable). lambda_nm in nanometres.
public float cauchyIOR(float base_ior, float cauchy_c, float lambda_nm) {
    float lambda_um = lambda_nm * 0.001;             // nm -> micrometre
    float l2        = lambda_um * lambda_um;
    // d-line reference (Fraunhofer d, He 587.6 nm) where base_ior is defined.
    const float kLdRefUm = 0.5876;
    float ld2 = kLdRefUm * kLdRefUm;
    float B   = base_ior - cauchy_c / ld2;           // anchor curve at d-line
    return B + cauchy_c / l2;
}

// Convert a Cauchy C-coefficient to an Abbe number V_d (and vice versa) so a
// material can be authored either way. Abbe V_d = (n_d - 1) / (n_F - n_C),
// the standard glass dispersion figure (BK7 ~ 64.2). The F and C Fraunhofer
// lines are 486.1 nm and 656.3 nm. Given base n_d this maps V_d -> C:
//   n_F - n_C = C*(1/lF^2 - 1/lC^2)  =>  C = (n_d - 1) / (V_d * k)
// where k = (1/lF^2 - 1/lC^2) in micrometre^-2. Higher V_d = LESS dispersion.
public float abbeToCauchyC(float base_ior, float abbe_vd) {
    if (abbe_vd <= 0.0) return 0.0;                  // 0 / negative = achromatic
    const float kLfUm = 0.4861, kLcUm = 0.6563;
    float k = (1.0 / (kLfUm * kLfUm)) - (1.0 / (kLcUm * kLcUm));
    return (base_ior - 1.0) / (abbe_vd * k);
}

// Wyman, Sloan & Shirley 2013, "Simple Analytic Approximations to the CIE
// XYZ Color Matching Functions" (JCGT). Single-lobe piecewise-Gaussian fits
// to the CIE 1931 2-deg observer. `t` here is the squared, branch-scaled
// argument; the helper returns exp(-0.5 t^2) with the paper's left/right
// sigma split folded into the caller. lambda in nanometres.
float wymanGauss(float x, float mu, float sigma1, float sigma2) {
    float t = (x - mu) * ((x < mu) ? sigma1 : sigma2);
    return exp(-0.5 * t * t);
}

// CIE 1931 X-bar / Y-bar / Z-bar at wavelength `lambda_nm` (nanometres) via
// the Wyman multi-lobe Gaussian fit. X-bar is a two-lobe sum (the curve has
// a small secondary bump in the blue); Y-bar and Z-bar are single lobes.
// Returns un-normalised tristimulus weights -- the path tracer folds in the
// band-average normaliser separately so a flat spectrum reconstructs white.
public float3 cieXYZFit(float lambda_nm) {
    float x = 1.056 * wymanGauss(lambda_nm, 599.8, 0.0264, 0.0323)
            + 0.362 * wymanGauss(lambda_nm, 442.0, 0.0624, 0.0374)
            - 0.065 * wymanGauss(lambda_nm, 501.1, 0.0490, 0.0382);
    float y = 0.821 * wymanGauss(lambda_nm, 568.8, 0.0213, 0.0247)
            + 0.286 * wymanGauss(lambda_nm, 530.9, 0.0613, 0.0322);
    float z = 1.217 * wymanGauss(lambda_nm, 437.0, 0.0845, 0.0278)
            + 0.681 * wymanGauss(lambda_nm, 459.0, 0.0385, 0.0725);
    return float3(x, y, z);
}

// Linear sRGB (Rec.709 primaries, D65) from CIE XYZ. Standard matrix; the
// caller keeps the result in linear space so it composes with the rest of
// the path tracer's linear-HDR pipeline (ACES tonemap runs downstream).
public float3 xyzToLinearSRGB(float3 xyz) {
    float r =  3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
    float g = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
    float b =  0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
    return float3(r, g, b);
}

// Per-wavelength linear-sRGB weight for the hero-wavelength estimator,
// normalised so the expected weight over a uniform draw across [380,730] is
// neutral (1,1,1). With that normaliser, a wavelength-INDEPENDENT path (no
// dispersion, RGB albedos constant across lambda) reconstructs exactly the
// non-spectral RGB image in expectation -- spectral mode only diverges from
// the RGB path where the IOR actually varies with lambda. The per-channel
// reciprocal-of-band-mean constants below are the integral of
// xyzToLinearSRGB(cieXYZFit(lambda)) over [380,730] / band width, precomputed
// once (Riemann sum, fine 1 nm steps) so the GPU pays three multiplies, not a
// loop. If the fit lobes above ever change, re-run the same Riemann sum over
// the new lobes and replace kInvBandMean so E[weight] stays (1,1,1).
public float3 spectralWeightSRGB(float lambda_nm) {
    float3 rgb = xyzToLinearSRGB(cieXYZFit(lambda_nm));
    // Reciprocal of each channel's band-mean (see derivation note above).
    // Band means over [380,730] (Riemann sum, 1 nm steps over the exact fit
    // + sRGB matrix below): R = 0.366194, G = 0.290365, B = 0.277250. The
    // reciprocals make E[weight] = (1,1,1) so a flat spectrum -> white.
    const float3 kInvBandMean = float3(2.730792, 3.443936, 3.606855);
    return rgb * kInvBandMean;
}
// --- end -------------------------------------------------------------------

// Ray-sphere intersection. Returns the nearer non-self-intersection t
// (>1e-3 epsilon to avoid stuck rays grazing their own surface).
public bool intersectSphere(float3 ro, float3 rd, float3 c, float rad, out float t) {
    float3 oc = ro - c;
    float b   = dot(oc, rd);
    float k   = dot(oc, oc) - rad * rad;
    float h   = b * b - k;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > 1e-3) ? t0 : t1;
    return t > 1e-3;
}

// Ray-plane intersection.  `n` is the plane normal, `d` the signed
// distance from origin (plane: n . p + d = 0).  Returns false for
// nearly-parallel rays (no near-zero division).
public bool intersectPlane(float3 ro, float3 rd, float3 n, float d, out float t) {
    float denom = dot(rd, n);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, n) + d) / denom;
    return t > 1e-3;
}

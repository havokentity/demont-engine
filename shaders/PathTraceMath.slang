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

// --- Scale-relative ray epsilons (issue #256, planetary P2) ----------------
//
// Every ray epsilon in this engine used to be the absolute constant 1e-3 m
// -- a 1 mm normal offset on 17 secondary-ray origins, a 1 mm `TMin` on
// the mesh query, a 1 mm near-root cut inside the intersectors below.  A
// single constant is wrong in BOTH directions the moment the scene stops
// being room-sized, and both ends bite on a planet:
//
//   TOO SMALL AT RANGE.  The hit point is ro + rd*t and its representation
//   error is ~2^-24 * max(|ro|, t).  At the horizon from ISS altitude
//   (t = 2.2e6 m) that is 0.13 m -- 130x the offset -- so every distant
//   surface shadow-acnes.
//
//   TOO LARGE UP CLOSE.  At t = 0.05 m the numerical error is 3 nm and a
//   1 mm offset visibly detaches the contact shadow.
//
// The replacement is two terms with different jobs, so both are needed:
//
// TERM 1 -- REPRESENTATIONAL, ptOffsetRay.  Wachter & Binder, Ray Tracing
// Gems ch. 6.  Offsets the position by a fixed number of ULPs OF ITS OWN
// COORDINATE, done as integer arithmetic on the exponent+mantissa field,
// so the step is exactly one representable neighbourhood at whatever
// magnitude the coordinate happens to have.  Nothing to tune and nothing
// that scales wrongly: it is the same offset in ULPs at 1 m and at 1e5 m.
//
//   Its published constants (256 ULPs; 1/65536 m near the origin, where
//   ULPs become smaller than any geometry) are calibrated for coordinates
//   of roughly [1e-2, 1e5] m.  THAT RANGE IS NOT AN ASSUMPTION HERE, IT IS
//   A GUARANTEE: planetary P1 (#255) packs every GPU-facing coordinate
//   relative to the camera anchor, and `r_origin_rebase_radius` (default
//   65536 m) bounds how far the anchor may lag the eye.  Shader-frame
//   coordinates are therefore bounded by that radius plus the trace
//   distance, which is exactly the band Wachter-Binder is valid over.
//   This is why P2 depends on P1 rather than merely following it: the
//   coordinate scheme is what makes this offset correct.
//
// TERM 2 -- GEOMETRIC, ptConeEps.  ptOffsetRay cannot fix acne whose cause
// is that the SHADED surface is not the TRACED surface: a normal map's
// shading normal diverging from the geometric one, or a displaced surface
// (oceanRayMarch) whose shaded height is not the height that was hit.
// That error is bounded by the sub-pixel geometry the ray never resolved,
// i.e. by the ray-cone footprint -- which #248 already carries: cone_width
// at a hit IS the pixel footprint there, in metres, in the same units as
// the offset.  A quarter of it is sub-pixel BY CONSTRUCTION at every scale,
// which is the invariant an absolute constant cannot have: the shadow can
// never visibly detach from its caster, because the detachment is smaller
// than the pixel that would show it.
//
// The two are combined by displacing along the normal, which is a max in
// everything but syntax -- they differ by orders of magnitude and only one
// is ever the active term.  Near the camera the cone is sub-millimetre and
// ptOffsetRay dominates; at range the cone dominates and the ULP step is
// noise beside it.

// Ray-cone state (RT Gems 2019 ch. 20), carried alongside a ray so every
// function that needs a scale has one.  `w` is the footprint DIAMETER at
// the ray's own origin and `spread` its growth per metre, so the footprint
// at distance t along the ray is w + spread*t.  A camera ray starts at
// w = 0 with spread = 2*tan(fovY/2)/height -- one pixel per unit distance.
public struct PtRayCone {
    float w;
    float spread;
};

public PtRayCone ptConeMake(float w, float spread) {
    PtRayCone c; c.w = w; c.spread = spread; return c;
}

// The cone advanced to distance t: the same beam, re-anchored.
public PtRayCone ptConeAdvance(PtRayCone c, float t) {
    return ptConeMake(c.w + c.spread * max(t, 0.0), c.spread);
}

// The footprint DIAMETER at distance t along the ray -- the same
// arithmetic as ptConeAdvance's width lane, exposed as a scalar because
// PtRayCone's fields are module-private and a caller outside this module
// cannot read them. Planetary P4 (#258) needs it at the terrain hit, where
// the sub-footprint slope variance is folded into microfacet roughness and
// the fold has to know the footprint before the bounce loop advances the
// cone.
public float ptConeWidthAt(PtRayCone c, float t) {
    return c.w + c.spread * max(t, 0.0);
}

// Wachter & Binder, "A Fast and Robust Method for Avoiding
// Self-Intersection" (Ray Tracing Gems ch. 6), verbatim.
//
// The integer add walks `int_scale` representable floats along +n, so the
// displacement is the coordinate's own ULP times a constant rather than a
// fixed distance.  The sign flip is because float32 is sign-magnitude:
// making a negative coordinate more negative means ADDING to its magnitude
// bits, so the integer offset has to be negated there.
//
// Below `origin` the ULP is smaller than any geometry this renderer can
// resolve, so a fixed `float_scale` step takes over -- that branch is the
// only absolute constant left, and it is 1/65536 m at coordinates under
// 1/32 m, i.e. still four orders below the smallest feature.
public float3 ptOffsetRay(float3 p, float3 n) {
    const float kOrigin     = 1.0 / 32.0;
    const float kFloatScale = 1.0 / 65536.0;
    const float kIntScale   = 256.0;
    int3   of_i = int3(kIntScale * n);
    float3 p_i  = asfloat(asint(p) + select(p < float3(0.0), -of_i, of_i));
    return select(abs(p) < float3(kOrigin), p + kFloatScale * n, p_i);
}

// The geometric term, in metres: a quarter of the footprint, widened as
// the outgoing ray lies down toward the surface.
//
// 0.25 is the sub-pixel budget -- a quarter of a pixel of detachment is
// under the reconstruction filter at every scale, which is the whole
// point of measuring the offset in footprints rather than in metres.
//
// The grazing widening is bounded rather than open, and the bound is
// derived from the same invariant: an offset d along the normal displaces
// the contact shadow by d*cot(phi) for a light phi above the surface, so
// an UNBOUNDED 1/cos would trade acne for peter-panning one-for-one and
// lose the invariant it was protecting.  Requiring the offset itself to
// stay within one whole footprint, 0.25*w/c <= w, gives c >= 0.25 exactly.
// So the widening runs from 1x at normal incidence to 4x at 76 degrees off
// the normal and stops there.
public float ptConeEps(float cone_w, float cos_theta) {
    return 0.25 * cone_w / max(cos_theta, 0.25);
}

// A secondary ray's origin: the representational step, then the geometric
// one along the same normal.  `dir` is where the ray is going, which is
// what sets the grazing widening.
public float3 ptRayOrigin(float3 p, float3 n, PtRayCone cone, float3 dir) {
    return ptOffsetRay(p, n) + n * ptConeEps(cone.w, abs(dot(n, dir)));
}

// The near cut, in metres along the ray -- what `1e-3` used to be at
// `r.TMin`, at the mesh triangle test, and inside the intersectors below.
//
// Two terms max'd, for the same reasons the origin offset has two.  The
// cone term is the same quarter-footprint budget: a hit closer than a
// quarter pixel cannot be distinguished from the origin's own surface.
// The representational term is ptOffsetRay's displacement expressed as a
// distance -- kIntScale ULPs of the largest origin coordinate, 256*2^-24
// relative -- so a ray whose origin was NOT offset (the camera ray, or an
// origin carried through a refraction) still cannot re-hit the surface it
// left through pure rounding.
//
// Both are zero-at-zero and scale-free.  A camera ray one metre from the
// anchor gets 1.5e-5 m; the same ray 2.2e6 m out gets 0.034 m, which is
// what the 1 mm constant could not do.
public float ptRayTMin(float3 ro, PtRayCone cone) {
    const float kIntScaleRel = 256.0 / 16777216.0;      // 256 ULPs, relative
    float m = max(max(abs(ro.x), abs(ro.y)), abs(ro.z));
    return max(0.25 * cone.w, kIntScaleRel * m);
}
// --- end scale-relative ray epsilons ---------------------------------------

// --- Planet-scale ray-sphere intersection (issue #254) ---------------------
//
// `dot(oc, oc) - rad * rad` and `-b - sqrt(b*b - k)` both subtract two
// nearly-equal large quantities whenever the ray origin sits near a large
// sphere's surface, and BOTH lose the same ~eps*rad of altitude.  At the
// Earth's radius one float32 ULP of |oc|^2 is 4.8e6 m^2, so an eye at
// h = 1.7 m -- where the true k = h(2R + h) = 2.2e7 m^2 -- keeps only
// three significant digits, and t comes back with ~0.38 m of noise.
// Rebasing the camera does not help: the cancellation is between |oc|^2
// and rad^2, and both are inherently large when the *sphere* is.
//
// Fixing only one of the two subtractions buys nothing -- each contributes
// ~eps*rad on its own -- so the substitution k = h(2R + h) that the
// planetary literature quotes is not a fix either: recovering h as
// length(oc) - rad is itself a cancelling subtraction that reintroduces
// exactly the error it was meant to remove.  The information is present
// in the float32 inputs (ro, c and rad are exact); extracting it needs
// arithmetic wider than float32, not a rearrangement of it.
//
// So: accumulate the cancelling sums in a two-word int32 fixed-point
// accumulator (~47 bits, enough for the 46-bit products), then take the
// root that does not cancel and get its partner from the product of roots
// (Vieta).  Integers are the load-bearing choice, not an implementation
// detail -- see ptFixedAdd.
//
// Small spheres take the historic expression verbatim.  Every sphere the
// renderer has ever traced is metres across, where the naive form is
// already good to a micron, so gating keeps every existing golden
// bit-identical instead of trading pixels for precision nobody uses.

// Radius above which the naive expression loses more than a millimetre of
// altitude.  Its radial error is ~eps*rad with eps = 2^-24, so 1 mm is
// reached at rad = 1e-3 * 2^24 = 16777 m; the constant is the power of
// two just below that, so the crossover is conservative and exact.
static const float kPtStableSphereRadius = 16384.0;

// Largest-operand exponent at which the fixed-point accumulators below
// still reach a FINITE float32 result.  One constant, shared by the
// ray-sphere and the altitude kernels, because they accumulate the same
// shape of sum and the copy is what went wrong (issue #275).
//
// The bound is on the RANGE OF THE TOTAL, not on the exponent field
// ptPow2 has to construct.  Confusing the two is the whole of #275 and it
// is worth a decade of exponents.  With m the largest operand and
// E = ptFloatExp(m), so m < 2^(E+1):
//
//   ptPowerOfPoint    |oc|^2 - rad^2       <= 3m^2 + m^2            =  4m^2
//   ptPowerOfPointAt  |p|^2 - 2 p.c + |c|^2 - rad^2
//                                          <= 3m^2 + 6m^2 + 3m^2 + m^2
//                                                                   = 13m^2
//
// The looser of the two is 13m^2 < 2^(2E+6), so finiteness needs
// 2E + 6 <= 128, i.e. E <= 61.  The constant sits one below that: the
// same E rides the accumulator's up_hi scale as well as the terms, and
// 2^60 m is 122 light-years, so there is nothing to buy by shaving it.
//
// The read doubles as a finiteness test at no cost: an infinity or a NaN
// among the operands reports exponent 128 and so routes to the naive
// forms rather than into the accumulator's integer arithmetic.
static const int kPtAccumMaxExp = 60;

// Unbiased exponent of |v|.  Zero and denormals report -127, which the
// callers below treat as "too small to matter", the correct answer here.
int ptFloatExp(float v) { return int((asuint(v) >> 23) & 0xFFu) - 127; }

// 2^n, exactly.  Built from the exponent field rather than exp2() so it is
// exact and so no fast-math approximation of exp2 can perturb the scale.
// Requires n in [-126, 127]; the intersectSphere gate guarantees that.
float ptPow2(int n) { return asfloat(uint(n + 127) << 23); }

// Two-word fixed-point accumulator for a sum whose terms are individually
// ~2^e but whose total is many orders of magnitude smaller.
//
// The words are int32 ON PURPOSE.  Every float-only compensated-summation
// scheme (Knuth's two-sum, Dekker's two-product, Kahan) is defined by an
// expression that is algebraically zero, so a compiler licensed to
// reassociate is licensed to fold it away -- and Metal compiles this
// module with its default (fast) math mode, which does exactly that:
// measured on an M4 Max, a Dekker/two-sum implementation of the code
// below returns the *naive* answer.  Integer addition has no such
// licence, so the cancellation survives every math mode intact.
//
// hi counts units of 2^(e-23), lo units of 2^(e-47).  Terms are bounded by
// 2^(e+1), so each contributes at most 2^24 to hi and 2^24 to lo, and the
// four-term sums stay far inside int32.  Residue below 2^(e-47) is dropped,
// which bounds the total error at a fraction of one ULP of the result.
struct PtFixedSum {
    int   hi;
    int   lo;
    float up_hi, up_lo;   // 2^(e-23), 2^(e-47)
    float dn_hi, dn_lo;   // reciprocals of the above
};

PtFixedSum ptFixedBegin(int e) {
    int sh = e - 23;
    int sl = e - 47;
    PtFixedSum a;
    a.hi = 0;
    a.lo = 0;
    a.up_hi = ptPow2(sh);  a.dn_hi = ptPow2(-sh);
    a.up_lo = ptPow2(sl);  a.dn_lo = ptPow2(-sl);
    return a;
}

// `ih` is exact (|x| <= 2^(e+1) so |ih| <= 2^24, and int <-> float is
// exact there); the remainder is a Sterbenz-exact subtraction of two
// nearly-equal floats.  Neither feeds a float add chain, so there is
// nothing for reassociation to get hold of.
void ptFixedAdd(inout PtFixedSum a, float x) {
    int   ih = int(x * a.dn_hi);
    float r  = x - float(ih) * a.up_hi;
    a.hi += ih;
    a.lo += int(r * a.dn_lo);
}

// Accumulate sgn * x * y with the 46-bit product formed EXACTLY.  Masking
// the low 12 mantissa bits splits each operand into halves of <= 12
// significant bits, so all four partial products fit a float32 mantissa
// with nothing to round away.  The mask is an integer op, so this split
// also survives fast math (verified on device).
void ptFixedAddProduct(inout PtFixedSum a, float x, float y, float sgn) {
    float xh = asfloat(asuint(x) & 0xFFFFF000u), xl = x - xh;
    float yh = asfloat(asuint(y) & 0xFFFFF000u), yl = y - yh;
    ptFixedAdd(a, sgn * (xh * yh));
    ptFixedAdd(a, sgn * (xh * yl));
    ptFixedAdd(a, sgn * (xl * yh));
    ptFixedAdd(a, sgn * (xl * yl));
}

float ptFixedTotal(PtFixedSum a) {
    return float(a.hi) * a.up_hi + float(a.lo) * a.up_lo;
}

// |oc|^2 - rad^2 -- the power of the point with respect to the sphere, and
// the quantity the naive form destroys.  Correctly rounded in practice:
// the accumulator resolves ~24 bits below the float32 result's own ULP.
//
// Public because altitude wants it too (issue #271): every "how far above
// the surface" query in the shader is this quantity divided by |oc| + rad.
public float ptPowerOfPoint(float3 oc, float rad) {
    float m = max(max(abs(oc.x), abs(oc.y)), max(abs(oc.z), rad));
    PtFixedSum a = ptFixedBegin(2 * ptFloatExp(m) + 1);
    ptFixedAddProduct(a, oc.x, oc.x,  1.0);
    ptFixedAddProduct(a, oc.y, oc.y,  1.0);
    ptFixedAddProduct(a, oc.z, oc.z,  1.0);
    ptFixedAddProduct(a, rad,  rad,  -1.0);
    return ptFixedTotal(a);
}

// dot(a, b) without the cancellation.  A grazing ray at planetary radius
// sums three ~R terms down to a few kilometres, so the plain dot product
// carries ~eps*R = 0.4 m of absolute error -- which lands directly on t
// for horizon rays, the exact case sub-metre terrain cares about.
//
// Public for the same reason as ptPowerOfPoint (issue #271): the hoisted
// ray-march altitude below needs (ro - c) . rd at full precision.
public float ptDotExact(float3 a, float3 b) {
    float m = max(max(abs(a.x), abs(a.y)), abs(a.z))
            * max(max(abs(b.x), abs(b.y)), abs(b.z));
    if (!(m > 0.0)) { return 0.0; }
    PtFixedSum s = ptFixedBegin(ptFloatExp(m) + 1);
    ptFixedAddProduct(s, a.x, b.x, 1.0);
    ptFixedAddProduct(s, a.y, b.y, 1.0);
    ptFixedAddProduct(s, a.z, b.z, 1.0);
    return ptFixedTotal(s);
}

bool ptIntersectSphereLarge(float3 oc, float3 rd, float rad, float t_min,
                            out float t) {
    float b = ptDotExact(oc, rd);
    float k = ptPowerOfPoint(oc, rad);
    // fma rounds once, on the *result*, so the discriminant keeps full
    // relative precision even where b*b and k nearly cancel (grazing).
    float h = fma(b, b, -k);
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    // Vieta / "citardauq": -b -/+ h subtracts two ~rad quantities for a
    // near-radial ray and throws away everything the accumulator just
    // bought.  Evaluate the root whose terms reinforce, then recover its
    // partner from the product of roots (= k).  This is the same
    // rearrangement Ray Tracing Gems ch. 7 uses; the accumulated k above
    // is what makes it worth having, since t = k/q inherits k's error
    // directly and RTG leaves k in its naive form.
    float q  = (b >= 0.0) ? -(b + h) : (h - b);
    float ta = q;
    float tb = (q != 0.0) ? k / q : 0.0;
    float t0 = min(ta, tb), t1 = max(ta, tb);
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

// The historic expression with the geometry shifted down by a power of
// two first, so that nothing it squares can leave float32 range.
//
// The quadratic |oc + t rd|^2 = rad^2 is homogeneous of degree one in t,
// so solving it for (oc*2^-s, rad*2^-s) and shifting the root back up by
// 2^s returns the SAME root.  Shifting by a power of two is exact -- every
// operand keeps its mantissa, so each product, sum and sqrt rounds exactly
// where it would have rounded unscaled -- which makes this line for line
// the answer the naive expression would give if float32 had unlimited
// exponent range, and not an approximation of it.
//
// After the shift |oc| <= 2 sqrt(3) and rad <= 2, so the sum of squares is
// at most 16 however large the geometry was.  (The factor of two is the
// ext_exp = 127 case, where 2^-127 is not a normal float and the shift
// has to stop one short.)
//
// It buys FINITENESS, not accuracy: the cancellation #254 is about is
// still there, so t carries the historic ~eps*rad error.  Accuracy above
// kPtAccumMaxExp is not on the table anyway -- 2^60 m is 122 light-years,
// where a float32 t cannot resolve a kilometre.
//
// The shift back up can itself overflow, when the true t is not a
// representable float32.  An infinity is the honest answer there, and the
// property that matters is that it is an infinity and not a NaN.
bool ptIntersectSphereScaled(float3 oc, float3 rd, float rad, int ext_exp,
                             float t_min, out float t) {
    int   s  = min(ext_exp, 126);
    float dn = ptPow2(-s), up = ptPow2(s);
    float3 o = oc * dn;
    float  r = rad * dn;
    float b  = dot(o, rd);
    float k  = dot(o, o) - r * r;
    float h  = b * b - k;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = (-b - h) * up, t1 = (-b + h) * up;
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

// Ray-sphere intersection. Returns the nearer non-self-intersection t,
// where "self" is anything nearer than `t_min` -- ptRayTMin above, no
// longer the absolute 1e-3 this used to bake in (issue #256).
//
// FINITE IN, FINITE OUT (issue #275), with no range carve-out: this runs
// per bounce and a NaN t poisons a whole path, not one sample.  Three
// bodies, chosen by one exponent read:
//
//   rad > kPtStableSphereRadius, ext_exp <= kPtAccumMaxExp
//       the accumulated solve, sub-millimetre at planetary radius --
//       what #254 was about.
//   ext_exp > kPtAccumMaxExp
//       the historic expression, shifted so it cannot overflow.
//   otherwise
//       the historic expression verbatim, so every sphere the renderer
//       actually traces keeps its exact bits and every golden stays put.
//
// The gate used to read `ext_exp <= 74`, which bounds the exponent field
// ptPow2 has to CONSTRUCT and not the range of the accumulated result --
// and |oc|^2 - rad^2 leaves float32 an entire decade of exponents earlier.
// Measured on that code in a general orientation: every ext_exp in
// [64, 74] took the accumulated path, overflowed k to an infinity and
// returned a non-finite t out of the k/q root, while everything above 74
// fell through to a naive expression that overflowed on its own.  Both
// windows are closed here, and the constant that closes the first is now
// the one the altitude kernels already used rather than a second copy.
public bool intersectSphere(float3 ro, float3 rd, float3 c, float rad,
                            float t_min, out float t) {
    float3 oc = ro - c;
    // Above kPtStableSphereRadius the naive expression below is worse than
    // a millimetre; take the accumulated path instead.
    float m = max(max(abs(oc.x), abs(oc.y)), max(abs(oc.z), rad));
    int ext_exp = ptFloatExp(m);
    if (rad > kPtStableSphereRadius && ext_exp <= kPtAccumMaxExp) {
        return ptIntersectSphereLarge(oc, rd, rad, t_min, t);
    }
    // ext_exp == 128 is an infinity or a NaN in oc or rad: not in the
    // domain, and the shift cannot help it, so it goes to the historic
    // line where it yields the same kind of garbage it always did rather
    // than the 0 * inf the shift would make of it.
    if (ext_exp > kPtAccumMaxExp && ext_exp <= 127) {
        return ptIntersectSphereScaled(oc, rd, rad, ext_exp, t_min, t);
    }
    float b   = dot(oc, rd);
    float k   = dot(oc, oc) - rad * rad;
    float h   = b * b - k;
    if (h < 0.0) { t = 0.0; return false; }
    h = sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

// Ray-plane intersection.  `n` is the plane normal, `d` the signed
// distance from origin (plane: n . p + d = 0).  Returns false for
// nearly-parallel rays (no near-zero division), and for hits nearer than
// `t_min` (issue #256; this too used to bake in an absolute 1e-3).
//
// This has the same shape of cancellation as intersectSphere -- dot(ro, n)
// tends to -d as the origin approaches the plane -- but not the same
// severity: the subtraction itself is Sterbenz-exact, so the error is just
// dot()'s own ~eps*|ro|, and it only bites for a plane whose |d| is large
// enough to make that a real distance.  Every plane the renderer has is a
// floor near the origin, so it is left alone deliberately; a planet-scale
// plane would need ptDotExact above.
public bool intersectPlane(float3 ro, float3 rd, float3 n, float d,
                           float t_min, out float t) {
    float denom = dot(rd, n);
    if (abs(denom) < 1e-4) { t = 0.0; return false; }
    t = -(dot(ro, n) + d) / denom;
    return t > t_min;
}

// --- Altitude above a sphere at planetary radius (issue #271) --------------
//
// `length(p - c) - rad` is the intersectSphere defect in its smallest
// form.  At the Earth's radius it differences two values that agree to
// seven digits and keeps ~0.5 m of noise -- measured below -- on a
// quantity that then goes into exp(-h / 1500) (Mie) and exp(-h / 8000)
// (Rayleigh), and that the altitude-indexed LUTs of #257 and the terrain
// LOD of #258 will read directly.
//
// Run the identity |oc|^2 - rad^2 = (|oc| - rad)(|oc| + rad) in the
// direction that AVOIDS the subtraction instead of creating it:
//
//     h = (|oc|^2 - rad^2) / (|oc| + rad)
//
// -- an exact numerator over a denominator that is a sum of two
// non-negative quantities and so cannot cancel at all.  The denominator
// only ever needs relative accuracy: an error there scales h instead of
// offsetting it, which is why ~2R can sit in it harmlessly.
//
// THE SUBSTITUTION IS NOT ENOUGH ON ITS OWN.  `oc = p - c` must never be
// formed: with the planet centre at (0, -R, 0), `p.y - c.y` is `p.y + R`,
// an addition into a magnitude whose ULP is 0.5 m, so materialising oc
// quantises the altitude to +/- 0.25 m BEFORE any of the above runs.
// Measured over a 100 km ground sweep at R = 6.371e6: naive 0.50 m,
// exact-k-over-|oc|+R on a materialised oc 0.25 m -- half the defect,
// still nowhere near what p carries.  ptPowerOfPointAt therefore
// accumulates
//
//     |p - c|^2 - rad^2  =  |p|^2 - 2 p.c + |c|^2 - rad^2
//
// from p, c and rad directly, so no float32 subtraction happens anywhere
// and the answer lands within a few ULPs of the returned altitude itself
// (3.3e-7 m at h = 1 m).  Same information argument as #254: p, c and rad
// are exact floats, so the altitude they determine is exact too; it just
// needs wider-than-float32 arithmetic to get at.

// Whether the accumulated path is both worth taking and inside the
// accumulator's domain.  Same crossover as intersectSphere -- the naive
// form's ~eps*rad error reaches a millimetre at 1e-3 * 2^24 -- so bodies
// at or below it keep the historic expression and every existing scene
// renders unchanged.
//
// The UPPER bound is kPtAccumMaxExp, derived at its definition above: it
// bounds the RANGE of the accumulated result rather than the exponent
// field ptPow2 has to build, which is a decade of exponents lower.
//
// Measured on the pre-fix code (issue #271 review): E in [64, 74] took the
// accumulated path, overflowed k to an infinity and returned a NaN out of
// k/d -- worse than the naive form it was replacing, and NaN in a march
// poisons a whole pixel.  intersectSphere carried the same window until
// #275; it now reads the same constant.

bool ptAltitudeStable(float3 p, float3 c, float rad) {
    if (!(rad > kPtStableSphereRadius)) { return false; }
    float m = max(max(max(abs(p.x), abs(p.y)), abs(p.z)),
                  max(max(abs(c.x), abs(c.y)), max(abs(c.z), rad)));
    return ptFloatExp(m) <= kPtAccumMaxExp;
}

// |oc| - rad, evaluated so that |oc|^2 cannot leave float32 range.
//
// Below the guard this is the historic expression, character for
// character, so every body at or below kPtStableSphereRadius -- which is
// every scene that has ever run -- keeps its exact bits.  Above it the
// vector is scaled by its largest component first: |oc/m| is at most
// sqrt(3), so the sum of squares cannot overflow however large oc is.
// That costs a divide on a path no real scene reaches, and buys a finite
// answer everywhere one exists.
//
// The upper half of the test (e <= 127, i.e. m itself finite) keeps an
// infinite oc on the historic line, where it yields an infinity rather
// than the NaN that inf/inf would produce in the scaled branch.  Garbage
// in stays the same kind of garbage out instead of getting worse.
float ptAltitudeNaive(float3 oc, float rad) {
    int e = ptFloatExp(max(max(abs(oc.x), abs(oc.y)), abs(oc.z)));
    if (e > kPtAccumMaxExp && e <= 127) {
        float m = max(max(abs(oc.x), abs(oc.y)), abs(oc.z));
        return m * length(oc / m) - rad;
    }
    return length(oc) - rad;
}

// |p - c|^2 - rad^2 with p - c never formed.  Ten exact products instead
// of ptPowerOfPoint's four; the cross terms carry a factor of two, so the
// accumulator is seeded one binade higher (2e+2, not 2e+1) to keep every
// term inside ptFixedAdd's 2^(e+1) bound.  That costs a factor of two in
// absolute resolution: 2^(e-47) is 0.5 m^2 at Earth radius, i.e. 4e-8 m
// of altitude, still a third of the float32 ULP of a one-metre altitude
// and so not the term that limits the answer.
//
// PRECONDITION: ptFloatExp(largest operand) <= kPtAccumMaxExp, i.e.
// what ptAltitudeStable tests.  Beyond it the result is not a finite
// float32 and the accumulator hands back an infinity.
public float ptPowerOfPointAt(float3 p, float3 c, float rad) {
    float m = max(max(max(abs(p.x), abs(p.y)), abs(p.z)),
                  max(max(abs(c.x), abs(c.y)), max(abs(c.z), rad)));
    PtFixedSum a = ptFixedBegin(2 * ptFloatExp(m) + 2);
    ptFixedAddProduct(a, p.x, p.x,  1.0);
    ptFixedAddProduct(a, p.y, p.y,  1.0);
    ptFixedAddProduct(a, p.z, p.z,  1.0);
    ptFixedAddProduct(a, p.x, c.x, -2.0);
    ptFixedAddProduct(a, p.y, c.y, -2.0);
    ptFixedAddProduct(a, p.z, c.z, -2.0);
    ptFixedAddProduct(a, c.x, c.x,  1.0);
    ptFixedAddProduct(a, c.y, c.y,  1.0);
    ptFixedAddProduct(a, c.z, c.z,  1.0);
    ptFixedAddProduct(a, rad, rad, -1.0);
    return ptFixedTotal(a);
}

// h = k / (rad + |oc|) given k = |oc|^2 - rad^2.  |oc| comes back out of k
// rather than out of a vector we would have to difference: |oc|^2 is
// exactly rad^2 + k, and fma rounds that once, on the result.  The max()
// only catches the accumulator's own sub-ULP residue at |oc| = 0 (p at the
// centre, where k = -rad^2 and the fma should land on exactly zero).
//
// PRECONDITION: k finite.  An infinite k makes d infinite too and k/d is
// then a NaN.  There is deliberately no guard for it: both callers gate
// their inputs to kPtAccumMaxExp, which bounds k well inside float32,
// and the tests assert that domain exhaustively.  A guard here would have
// to invent a return value for an input whose true altitude is not
// representable, which would hide a future caller's mistake instead of
// surfacing it -- and this is the per-sample function in a march, so it
// is the one place where a branch is not free.
public float ptAltitudeFromPower(float k, float rad) {
    float d = rad + sqrt(max(fma(rad, rad, k), 0.0));
    return (d > 0.0) ? (k / d) : 0.0;
}

// Signed altitude of p above the sphere (c, rad); negative below the
// surface, -rad at the centre, both by construction rather than by a
// branch -- k is negative there and the denominator stays positive.
//
// FINITE for every finite p, c and rad whose difference p - c is itself
// finite.  Above the gate that is ptAltitudeNaive's scaling; below it,
// the accumulator's domain bound.  (If p - c overflows, no finite
// altitude exists to return.)
public float ptAltitudeAboveSphere(float3 p, float3 c, float rad) {
    if (!ptAltitudeStable(p, c, rad)) { return ptAltitudeNaive(p - c, rad); }
    return ptAltitudeFromPower(ptPowerOfPointAt(p, c, rad), rad);
}

// Altitude along a ray, hoisted.  For a unit rd the power of the point at
// p(s) = ro + rd*s is a quadratic in s,
//
//     |p(s) - c|^2 - rad^2 = k0 + 2 s b0 + s^2
//
// with k0 and b0 constant along the ray, so a march pays the accumulator
// once and two fma per step -- no more per-step work than the length()
// it replaces.
//
// Hoisting is not just the cheap option, it is the ACCURATE one, because
// p(s) is then never materialised.  A march that forms ro + rd*s and takes
// its altitude is capped by the float32 resolution of that position, which
// is ULP(|p|) -- 0.5 m in any frame whose origin is not near the surface.
// Measured over a +/-90 degree elevation sweep, 300 km rays, altitudes
// clipped to the 0-20 km band the optical depth actually integrates:
//
//   frame                             naive     hoisted   per-step exact
//   world origin on the surface       0.87 m    7.1e-3 m    2.7e-3 m
//   world origin at the centre        0.66 m    6.8e-3 m    2.7e-1 m
//
// The second row is the one that decides it: the per-step form collapses
// to the naive form's error the moment the frame moves, and the frame is
// exactly what P1 (#255) is changing.
public struct PtRayAltitude {
    float3 ro, rd, c;
    float  rad;
    float  k0;        // |ro - c|^2 - rad^2, accumulated
    float  b0;        // (ro - c) . rd, accumulated
    bool   stable;
};

// `rd` MUST be unit length -- the s^2 coefficient above is |rd|^2.  Every
// caller in this engine already guarantees that for other reasons.
//
// b0 is taken from a materialised ro - c, and that is deliberate: its
// 0.25 m quantisation enters h as s*delta/rad ~ 4e-8*s, below the eps*s
// the ray parameterisation itself carries, so removing it would buy
// nothing.  The dot product still goes through the accumulator, because
// on a horizon ray three ~rad terms cancel down to kilometres and a plain
// dot() keeps only eps*rad of that -- worth 1.8x on the measured sweep
// once the planet frame stops being axis-aligned.
public PtRayAltitude ptRayAltitudeBegin(float3 ro, float3 rd, float3 c, float rad) {
    PtRayAltitude a;
    a.ro = ro; a.rd = rd; a.c = c; a.rad = rad;
    a.stable = ptAltitudeStable(ro, c, rad);
    a.k0 = a.stable ? ptPowerOfPointAt(ro, c, rad) : 0.0;
    a.b0 = a.stable ? ptDotExact(ro - c, rd)       : 0.0;
    return a;
}

// FINITE for every finite s for which ro + rd*s is finite.
//
// `s` is bounded HERE rather than at Begin because it is the one input
// that varies along the march, and s^2 leaves float32 above 2^64 -- which
// is not academic: this engine passes 1e30 as a "no limit" ray length in
// the cloud shadow march a few hundred lines away, and 1e30 squared is an
// infinity that k/d would turn into a NaN.  Measured on the pre-fix code,
// this function returned NaN from s = 1e20 upward.  The exponent read is
// the same one the gate uses, so an infinite or NaN s routes out too.
public float ptRayAltitudeAt(PtRayAltitude a, float s) {
    if (!a.stable || ptFloatExp(s) > kPtAccumMaxExp) {
        return ptAltitudeNaive(a.ro + a.rd * s - a.c, a.rad);
    }
    // Nested so each fma rounds once on its OWN result: the error is then
    // an ULP of the value being formed, not of the largest term.  That is
    // what keeps a ray descending back to the surface -- where k0, 2 s b0
    // and s^2 cancel by two orders -- as accurate as one going straight up.
    float k = fma(s, s, fma(2.0 * a.b0, s, a.k0));
    return ptAltitudeFromPower(k, a.rad);
}
// --- end altitude above a sphere -------------------------------------------

// --- Wave 10 bloom/bokeh: thin-lens aperture sampling --------------------
// Shirley-Chiu concentric-disk mapping: square (u1,u2) in [0,1]^2 -> a
// uniformly distributed point on the unit disk.  This is the common base
// for the round and polygonal apertures; verbatim the math the camera-ray
// block used inline before wave-10, so blades=0 stays bit-for-bit.
// Mirrored in StarsComposite.slang (self-contained TU -- the sun/moon
// bokeh must match this sampler exactly).
public float2 ptConcentricDisk(float u1, float u2) {
    float a = 2.0 * u1 - 1.0;
    float b = 2.0 * u2 - 1.0;
    float r, phi;
    if (abs(a) > abs(b)) {
        r = a;
        phi = (3.14159265358979 / 4.0) * (b / max(abs(a), 1e-6));
    } else {
        r = b;
        phi = (3.14159265358979 / 2.0) -
              (3.14159265358979 / 4.0) * (a / max(abs(b), 1e-6));
    }
    return r * float2(cos(phi), sin(phi));
}

// Area-uniform sample of a regular N-gon aperture (circumradius 1) rotated
// by rot_rad, drawn DIRECTLY from two canonical uniforms (u1,u2) in [0,1]^2
// rather than by warping a disk sample.
//
// Why not the old disk->polygon radial scaling (r' = r * cos(pi/N)/cos(theta
// folded))?  That map's Jacobian is poly_r(theta)^2, which varies with the
// sample angle, so a uniform disk density becomes a NON-uniform polygon
// density (PDF ~ 1/R(theta)^2).  The thin-lens camera path applies no
// importance weight to the lens sample, so the biased density makes the
// integrated bokeh brightness/structure physically wrong (the polygon fills
// preferentially toward the vertices vs the flats).  See PR #235.
//
// The unbiased construction is the standard triangle-fan: a regular N-gon
// is N congruent isoceles triangles sharing the aperture centre.  (a) pick
// one wedge uniformly, (b) uniform-area sample that triangle with the
// sqrt-barycentric parameterization, (c) the polygon orientation falls out
// of placing the vertices at angle rot_rad.  Each triangle has equal area
// and is chosen with probability 1/N, so the marginal density is uniform
// over the whole polygon -- an unbiased aperture sample.  The polygon keeps
// the old circumradius (1, vertices on the unit circle) and inradius
// cos(pi/N), so the bokeh SIZE is unchanged; only its interior density is
// corrected from biased to uniform.
//
// Reusing u1 for BOTH the wedge pick and the in-triangle radius (via the
// fractional part u1*N - k) keeps the sampler at two uniforms, matching the
// disk path's RNG draw count so the blades>=3 path doesn't shift the seed
// stream relative to blades=0.  The fractional part of a uniform is itself
// uniform on [0,1] and independent of the integer part, so this is exact,
// not an approximation.  blades is assumed >= 3 (the caller keeps the round
// disk for fewer).  Mirrored verbatim in StarsComposite.slang.
public float2 ptPolygonSample(float u1, float u2, int blades, float rot_rad) {
    float n = float(blades);
    float wedge = 6.28318530717958 / n;          // 2*pi / N
    // (a) Pick a wedge k in [0, N-1] uniformly, then recycle the fractional
    //     part as a fresh uniform for the in-triangle radius parameter.
    float kf = floor(u1 * n);
    float ku = u1 * n - kf;                       // fractional part, ~U[0,1]
    // (c) Two adjacent polygon vertices on the unit circumradius circle.
    //     V0 (the aperture centre) is the origin, so its barycentric term
    //     drops out of the blend below.
    float a0 = rot_rad + kf * wedge;
    float a1 = a0 + wedge;
    float2 V1 = float2(cos(a0), sin(a0));
    float2 V2 = float2(cos(a1), sin(a1));
    // (b) Uniform-area sample of triangle (0, V1, V2) via the sqrt-bary map:
    //     p = (1-su)*V0 + su*(1-u2)*V1 + su*u2*V2, with V0 = 0.
    float su = sqrt(ku);
    return su * ((1.0 - u2) * V1 + u2 * V2);
}

// Cat's-eye / optical vignetting.  Off-axis image points see an entrance
// pupil partly occluded by the rear elements, so the bokeh clips into a
// lemon shape squeezed toward frame centre.  `uv` is the pixel's
// normalised [-1,1] screen position (|uv| ~ image eccentricity), strength
// in [0,1].  We compress the lens-sample component along the radial-to-
// centre axis by a factor that grows with eccentricity, which is the
// physically-correct narrowing of the pupil off-axis.  strength 0 (or an
// on-axis pixel) returns ap unchanged, so the bokeh stays round there.
public float2 ptCatEye(float2 ap, float2 uv, float strength) {
    float ecc = saturate(length(uv));            // 0 at centre, ~1 at corner
    float squeeze = strength * ecc;
    if (squeeze < 1e-5) return ap;
    float2 radial = (ecc > 1e-5) ? (uv / length(uv)) : float2(1.0, 0.0);
    float2 tangent = float2(-radial.y, radial.x);
    // Decompose the lens sample into radial + tangential, narrow the radial
    // part (the axis along which the pupil is occluded). Up to ~70% squeeze
    // at the frame corner for strength 1 -- a strong but still readable
    // lemon, matching a fast lens wide open.
    float ar = dot(ap, radial);
    float at = dot(ap, tangent);
    ar *= (1.0 - 0.7 * squeeze);
    return radial * ar + tangent * at;
}
// --- end Wave 10 bloom/bokeh ---------------------------------------------

// --- Planetary P3 (#257): the medium, parameterised by body ---------------
//
// Everything below is one body's atmosphere expressed as data rather than
// as literals scattered through the kernels.  Before this, Earth was baked
// into hardcoded numbers in four files -- the Rayleigh scale height 8000.0
// appeared in PathTrace.slang twice, in CloudsRaymarch.slang once and in
// atmosphericTransmittance once; the Mie scale height and the Rayleigh
// sigma triple in two -- so "which body is this?" was not a question the
// code could be asked.  A second body would have been a rewrite; a struct
// is an afternoon, and that is the entire reason this is a struct with one
// constructor in it today.
//
// EVERY CONSTANT TRACES.  This project has already once had to replace
// fabricated sky coefficients with the genuine ArHosekSkyModel dataset
// (commit b2111dd), so an unsourced number here would be that decision
// made again.  Sources are named per field on ptAtmoEarth below.
//
// UNITS: metres and inverse metres throughout, 1 world unit = 1 metre.
// RGB triples are sampled at 680 / 550 / 440 nm, the convention Bruneton
// 2017 and Hillaire 2020 both tabulate against.
public struct PtAtmoBody {
    // Geometry.
    public float  ground_radius;      // R_g  [m]
    public float  top_radius;         // R_t  [m], top of the modelled shell

    // Rayleigh: molecular scattering.  No absorbing partner -- air
    // molecules do not absorb in the visible, which is why this species
    // has a scattering coefficient and nothing else.
    public float3 rayleigh_sigma_s;   // [m^-1] at r = R_g
    public float  rayleigh_scale_h;   // [m]

    // Mie: aerosol.  Both halves are real.  The engine had NO Mie
    // absorption at all before #257, which made haze a pure scatterer and
    // therefore too bright at every optical depth.
    public float3 mie_sigma_s;        // [m^-1] at r = R_g
    public float3 mie_sigma_a;        // [m^-1] at r = R_g
    public float  mie_scale_h;        // [m]
    public float  mie_g;              // Cornette-Shanks asymmetry

    // Ozone: absorption only, and NOT exponential in altitude -- it is a
    // stratospheric layer, so it gets a tent profile.  Its Chappuis band
    // absorbs in the red-orange, which is what keeps a twilight zenith
    // blue rather than brown once the slant path through the layer gets
    // long.  Absent from this engine entirely before #257.
    public float3 ozone_sigma_a;      // [m^-1] at the tent peak
    public float  ozone_center;       // [m]
    public float  ozone_half_width;   // [m]
};

// Earth.  `ground_radius` is a parameter rather than a literal because the
// engine already owns it as a cvar (r_planet_radius, default 6 371 008.8 m
// = the IUGG mean radius) and because a body's radius is the one field a
// user legitimately pokes.
//
// Sources, field by field:
//
//   rayleigh_sigma_s  Bodhaine, Wood, Dutton & Slusser 1999, J. Atmos.
//                     Oceanic Technol. 16:1854-1861 for the molecular
//                     cross-sections; the (5.802, 13.558, 33.100)e-6 m^-1
//                     RGB triple as tabulated in Hillaire 2020, "A
//                     Scalable and Production Ready Sky and Atmosphere
//                     Rendering Technique" (EGSR 2020).  The engine
//                     previously carried (5.8, 13.5, 33.1)e-6, the same
//                     numbers rounded to three figures; restored to the
//                     tabulated precision.
//   rayleigh_scale_h  8 000 m.  US Standard Atmosphere 1976 density
//                     profile as fit by Bruneton 2008 / Hillaire 2020.
//   mie_sigma_s       3.996e-6 m^-1, Hillaire 2020.  Grey, because
//                     aerosol scattering is only weakly wavelength
//                     dependent across the visible -- which is why haze
//                     is white and Rayleigh is blue.
//   mie_sigma_a       4.400e-6 m^-1, Hillaire 2020.  The pair gives a
//                     single-scattering albedo of 3.996/(3.996+4.400) =
//                     0.4760: this aerosol absorbs slightly more than it
//                     scatters.
//   mie_scale_h       1 200 m, Hillaire 2020.  The engine used 1 500 m,
//                     which puts 25% too much aerosol above every
//                     altitude.
//   mie_g             0.8, Hillaire 2020, evaluated through the Cornette
//                     & Shanks 1992 phase function (J. Opt. Soc. Am. A
//                     9(12):2069) rather than plain Henyey-Greenstein --
//                     Cornette-Shanks reduces exactly to the Rayleigh
//                     phase at g = 0, which HG does not, so one form
//                     covers both species consistently.
//   ozone_sigma_a     (0.650, 1.881, 0.085)e-6 m^-1 at the tent peak.
//                     Cross-sections from Serdyuchenko et al. 2014 /
//                     Gorshelev et al. 2014; the RGB triple as used by
//                     Bruneton 2017 and Hillaire 2020.  The GREEN channel
//                     being the largest is the Chappuis band, and it is
//                     the whole physical point of including ozone.
//   ozone_center /    25 000 m and 15 000 m.  US Standard Atmosphere 1976
//   ozone_half_width  ozone profile, as fit by Bruneton's 2017 reference
//                     implementation to a symmetric tent.
//   top_radius        R_g + 100 000 m, the Karman line.  Bruneton 2008
//                     uses +80 km and Hillaire 2020 +60 km; +100 km is
//                     the defensible physical boundary and costs nothing,
//                     since density there is ~1e-7 of sea level and
//                     contributes below one part in 1e6 of any optical
//                     depth.
public PtAtmoBody ptAtmoEarth(float ground_radius) {
    PtAtmoBody b;
    b.ground_radius    = ground_radius;
    b.top_radius       = ground_radius + 100000.0;
    b.rayleigh_sigma_s = float3(5.802e-6, 13.558e-6, 33.100e-6);
    b.rayleigh_scale_h = 8000.0;
    b.mie_sigma_s      = float3(3.996e-6, 3.996e-6, 3.996e-6);
    b.mie_sigma_a      = float3(4.400e-6, 4.400e-6, 4.400e-6);
    b.mie_scale_h      = 1200.0;
    b.mie_g            = 0.8;
    b.ozone_sigma_a    = float3(0.650e-6, 1.881e-6, 0.085e-6);
    b.ozone_center     = 25000.0;
    b.ozone_half_width = 15000.0;
    return b;
}

// Apply the engine's per-species user scales to a body.
//
// `mie_sigma_s_abs` is an ABSOLUTE sea-level Mie SCATTERING coefficient in
// m^-1 -- the engine's r_volumetric_density.  Absolute, not a multiplier,
// so a user asking for a hazy day states the haze rather than a factor on
// top of an invisible baseline.
//
// Which half of the pair the knob names is a real choice and it was made
// by measurement, not by taste.  Before #257 the engine had NO aerosol
// absorption at all: it used this one number as extinction in the
// transmittance integral AND as the scattering coefficient in the
// in-scatter march, which is the same as asserting a single-scattering
// albedo of 1.  Adding the real absorbing half therefore has to break one
// of those two readings.  Taking the knob as EXTINCTION preserves aerial
// perspective and halves every haze glow; taking it as SCATTERING
// preserves the glow and deepens aerial perspective.  Measured on
// clouds_godrays, the fixture that leans hardest on both (r_volumetric
// _density 0.008, r_volumetric_intensity 2.0): the extinction reading
// moves it by mean 25.1 L2, the scattering reading by 4.2.  Six times
// smaller, for identical physics -- so the knob names scattering, a user
// who set a haze level keeps the haze they set, and the correction shows
// up where it belongs, as somewhat deeper aerial perspective.
//
// The absorbing half moves with it, in the ratio the body declares --
// 4.400 / 3.996 for Earth.  That ratio is the aerosol's own
// single-scattering albedo (0.476), a property of the particles rather
// than of how many there are, so it is what stays fixed as loading is
// dialled up and down: raising this means MORE of the same aerosol, not a
// different aerosol.
//
// `rayleigh_scale` and `ozone_scale` are dimensionless multiples of the
// real column (1.0 = real Earth).  Neither exists to compensate for the
// other; a pair of knobs tuned against each other is exactly the failure
// #257 retires.
public PtAtmoBody ptAtmoScale(PtAtmoBody b, float rayleigh_scale,
                              float mie_sigma_s_abs, float ozone_scale) {
    b.rayleigh_sigma_s = b.rayleigh_sigma_s * max(rayleigh_scale, 0.0);
    float ss0 = b.mie_sigma_s.g;
    // Guarded so a body authored with a non-scattering (or absent)
    // aerosol cannot divide by zero; it then simply stays that way.
    float3 co_albedo = (ss0 > 0.0) ? (b.mie_sigma_a / ss0)
                                   : float3(0.0, 0.0, 0.0);
    float ss = max(mie_sigma_s_abs, 0.0);
    b.mie_sigma_s = float3(ss, ss, ss);
    b.mie_sigma_a = co_albedo * ss;
    b.ozone_sigma_a = b.ozone_sigma_a * max(ozone_scale, 0.0);
    return b;
}

// Ozone's vertical profile: a symmetric tent peaking at ozone_center and
// reaching zero one half-width either side.  Bruneton 2017's fit to the
// US Standard Atmosphere 1976 profile.
//
// Deliberately NOT an exponential.  Ozone is a stratospheric layer with
// almost none of it at the surface, so an exponential -- which is maximal
// at the ground -- would have exactly the wrong shape: it would redden a
// short vertical path (where real ozone does nothing) and under-absorb the
// long twilight slant path (where real ozone does everything).
public float ptAtmoOzoneProfile(PtAtmoBody b, float h) {
    return max(0.0, 1.0 - abs(h - b.ozone_center) / b.ozone_half_width);
}

// Per-altitude scattering and extinction: one evaluation, three species.
// The out-parameter shape is deliberate -- it makes "include a species in
// extinction but not in scattering" (or the reverse) unrepresentable at a
// call site, which is the bug class this signature exists to close.
public void ptAtmoCoefficients(PtAtmoBody b, float h,
                               out float3 sigma_s_rayleigh,
                               out float3 sigma_s_mie,
                               out float3 sigma_t) {
    float hc = max(h, 0.0);
    float dr = exp(-hc / b.rayleigh_scale_h);
    float dm = exp(-hc / b.mie_scale_h);
    float doz = ptAtmoOzoneProfile(b, hc);
    sigma_s_rayleigh = b.rayleigh_sigma_s * dr;
    sigma_s_mie      = b.mie_sigma_s * dm;
    sigma_t = sigma_s_rayleigh
            + (b.mie_sigma_s + b.mie_sigma_a) * dm
            + b.ozone_sigma_a * doz;
}

// Rayleigh phase function, 3/(16 pi) (1 + mu^2), normalised over the
// sphere.  `mu` is the cosine between the incident propagation direction
// and the outgoing one.
public float ptPhaseRayleigh(float mu) {
    return 0.0596831036594607 * (1.0 + mu * mu);   // 3 / (16 pi)
}

// Cornette & Shanks 1992 phase function.  Reduces exactly to the Rayleigh
// phase at g = 0, so a debug "isotropise the aerosol" does not silently
// change which physics is being modelled -- which plain Henyey-Greenstein
// (isotropic at g = 0) would.
public float ptPhaseCornetteShanks(float mu, float g) {
    float g2 = g * g;
    float num = 3.0 * (1.0 - g2) * (1.0 + mu * mu);
    float den = 8.0 * 3.14159265358979 * (2.0 + g2)
              * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5);
    return num / den;
}

// Optical depth along [ro, ro + rd*t] through the body's medium, by
// composite Simpson's rule on `steps` subintervals.
//
// PRECONDITION: `rd` is unit length.  It is the s^2 coefficient of the
// hoisted altitude quadratic as well as the integration measure, so a
// non-normalised direction corrupts both.
//
// Altitudes come from ptRayAltitudeAt, i.e. the hoisted cancellation-free
// form (#271): the quadratic is accumulated once per call and each sample
// costs two fma.  The naive alternative carries ~0.5 m of noise at
// planetary radius straight into exp(-h/1200), where 0.5 m is 0.04% of the
// scale height on every sample.
//
// Simpson is exact for cubics, and Rayleigh and Mie are smooth
// exponentials in a quadratic-in-s altitude, so 8 subintervals hold ~0.1%
// over the path lengths this engine integrates.  Ozone's tent is only
// piecewise linear: its two breakpoints (h = centre +/- half-width) are
// corners Simpson cannot see, so the ozone term carries a larger relative
// error than the other two.  Stated rather than hidden, and bounded: the
// vertical ozone optical depth is sigma_peak * half_width = 1.881e-6 *
// 15000 = 0.028 in green, so a few percent of the ozone term is a few
// parts in 1e4 of the total.  A transmittance LUT removes this entirely by
// integrating each (r, mu) once at build time with a finer rule; that is
// the structure this function is shaped to be replaced by.
public float3 ptAtmoOpticalDepth(PtAtmoBody b, float3 ro, float3 rd,
                                 float3 centre, float t, int steps) {
    if (!(t > 0.0)) return float3(0.0, 0.0, 0.0);
    int N = max(steps, 1);
    float dt = t / float(N);
    float oneSixth = 1.0 / 6.0;
    PtRayAltitude alt = ptRayAltitudeBegin(ro, rd, centre, b.ground_radius);
    float3 tau = float3(0.0, 0.0, 0.0);
    // The left-boundary extinction is cached from the previous
    // subinterval's right boundary, halving the exp() count across the
    // march -- the same boundary-caching the pre-#257 integral used.
    float3 sr_l, sm_l, st_prev;
    ptAtmoCoefficients(b, max(ptRayAltitudeAt(alt, 0.0), 0.0),
                       sr_l, sm_l, st_prev);
    for (int i = 0; i < N; ++i) {
        float s_mid   = (float(i) + 0.5) * dt;
        float s_right = (float(i) + 1.0) * dt;
        float3 sr_m, sm_m, st_mid;
        float3 sr_r, sm_r, st_right;
        ptAtmoCoefficients(b, max(ptRayAltitudeAt(alt, s_mid),   0.0),
                           sr_m, sm_m, st_mid);
        ptAtmoCoefficients(b, max(ptRayAltitudeAt(alt, s_right), 0.0),
                           sr_r, sm_r, st_right);
        tau += dt * oneSixth * (st_prev + 4.0 * st_mid + st_right);
        st_prev = st_right;
    }
    return tau;
}

// Both roots of the ray-sphere quadratic, on the same three-body gate
// intersectSphere uses.  intersectSphere returns only the nearer
// non-self root, and a shell needs both, so this exposes the solve
// rather than duplicating it: the accumulated path above
// kPtStableSphereRadius, the historic expression below it so every
// scene-scale sphere keeps its exact bits.
//
// Vieta ("citardauq") on the root pair: -b +/- h subtracts two ~rad
// quantities for a near-radial ray and throws away everything the
// accumulator just bought, so evaluate the root whose terms reinforce and
// recover its partner from the product of roots (= k).  Ray Tracing Gems
// ch. 7, with the accumulated k that makes it worth having.
public bool ptSphereRoots(float3 ro, float3 rd, float3 c, float rad,
                          out float t0, out float t1) {
    t0 = 0.0; t1 = 0.0;
    if (!(rad > 0.0)) return false;
    float3 oc = ro - c;
    float m = max(max(abs(oc.x), abs(oc.y)), max(abs(oc.z), rad));
    int ext_exp = ptFloatExp(m);
    float b, k;
    if (rad > kPtStableSphereRadius && ext_exp <= kPtAccumMaxExp) {
        b = ptDotExact(oc, rd);
        k = ptPowerOfPoint(oc, rad);
    } else {
        b = dot(oc, rd);
        k = dot(oc, oc) - rad * rad;
    }
    float h = fma(b, b, -k);
    if (h < 0.0) return false;
    h = sqrt(h);
    float q  = (b >= 0.0) ? -(b + h) : (h - b);
    float ta = q;
    float tb = (q != 0.0) ? k / q : 0.0;
    t0 = min(ta, tb);
    t1 = max(ta, tb);
    return true;
}

// Ray versus a spherical SHELL [r_inner, r_outer] about `centre`.
//
// Returns the parametric interval of the ray inside the shell, in front of
// the origin, clipped to [0, t_max].  False when the ray never enters it.
//
// This is what replaces the flat-slab intersections the cloud layer used,
// and the difference is the founding symptom of #51.  A slab is a pair of
// infinite planes, so at a ground camera a perfectly horizontal ray needs
// infinite horizontal distance to reach a layer above it -- clouds
// therefore never touched the horizon, they stopped at a hard line and
// left a gap of bare sky below.  On a shell the same ray exits through the
// top of the layer at a finite range, because the ground curves away
// underneath it, and the layer terminates AT the horizon.
//
// Rather than enumerate the origin cases (below the layer, inside it,
// above it), the interval is computed as "inside the outer sphere MINUS
// inside the inner sphere" and the NEAR component returned, which is
// correct for every origin by construction.  The far component -- the far
// side of the shell, beyond the body -- is deliberately dropped: it is
// occluded by the body for any origin outside the inner sphere, and
// returning it would have a ground camera marching clouds on the other
// side of the world.
//
// The roots come from ptSphereRoots, i.e. the stable solve at planetary
// radius (#254 / #275).  That matters more here than anywhere: at
// |oc| ~ 6.4e6 the naive discriminant loses 22% of k, which is ~0.38 m of
// altitude error, against a cloud shell only a few hundred metres thick.
public bool ptRayShell(float3 ro, float3 rd, float3 centre,
                       float r_inner, float r_outer, float t_max,
                       out float t_in, out float t_out) {
    t_in = 0.0; t_out = 0.0;
    if (!(r_outer > 0.0) || !(t_max > 0.0)) return false;
    float o0, o1;
    if (!ptSphereRoots(ro, rd, centre, r_outer, o0, o1)) return false;
    float lo = max(o0, 0.0);
    float hi = min(o1, t_max);
    if (!(hi > lo)) return false;
    float i0, i1;
    if (r_inner > 0.0 && ptSphereRoots(ro, rd, centre, r_inner, i0, i1)) {
        // The inner sphere carves [i0, i1] out of [lo, hi].  If the carve
        // starts after lo the near piece is [lo, i0]; if the origin is
        // already inside the inner sphere (i0 <= lo < i1) the ray emerges
        // at i1 and that becomes the new near bound.
        if (i1 > lo) {
            if (i0 > lo) { hi = min(hi, i0); }
            else         { lo = i1; }
        }
    }
    if (!(hi > lo)) return false;
    t_in = lo; t_out = hi;
    return true;
}
// --- end Planetary P3 -----------------------------------------------------

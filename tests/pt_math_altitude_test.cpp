// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for cancellation-free altitude above a sphere (issue #271),
// the second half of the kernel #254 / PR #267 introduced.
//
// WHAT BROKE
//
// `planetAltitude()` and the spherical optical-depth march both computed
// altitude as `length(p - c) - R`.  At the Earth's radius that differences
// two values agreeing to seven digits: one float32 ULP of |p - c| is
// 0.5 m, so the answer arrives with half a metre of noise -- and it then
// feeds exp(-h / 1500) for Mie and exp(-h / 8000) for Rayleigh, where a
// metre is 0.07% of the scale height on every one of the 17 Simpson
// samples.  Measured worst case over a 100 km ground sweep: 0.50 m.
//
// WHY THE OBVIOUS SUBSTITUTION IS NOT THE FIX
//
// The identity |oc|^2 - R^2 = (|oc| - R)(|oc| + R) gives
// h = ptPowerOfPoint(oc, R) / (|oc| + R): an exact numerator over a
// denominator that cannot cancel.  Applied to a materialised
// `oc = p - c`, that only halves the error -- to 0.25 m, pinned below.
// The reason is upstream of the arithmetic: with the centre at (0, -R, 0),
// `p.y - c.y` IS `p.y + R`, an addition into a magnitude whose ULP is
// 0.5 m, so oc is already a quantised version of a point that p described
// to the millimetre.  ptPowerOfPointAt therefore accumulates
// |p|^2 - 2 p.c + |c|^2 - R^2 from p, c and R directly and never forms a
// float32 difference at all.
//
// The ray march adds a second version of the same trap.  Evaluating the
// altitude at a materialised p(s) = ro + rd*s caps the answer at the
// float32 resolution of that position -- fine while the world origin sits
// on the surface, catastrophic (0.27 m, pinned below) the moment the
// frame moves, which is exactly what planetary P1 is doing.  Hoisting the
// quadratic |p(s) - c|^2 - R^2 = k0 + 2 s b0 + s^2 avoids materialising
// p(s) at all, and costs two fma per step instead of a length().
//
// WHY THIS FILE MIRRORS THE SHADER
//
// The fix lives in shaders/PathTraceMath.slang and there is no host entry
// point to call, so this file transcribes it -- same operations, same
// order, since the order is the whole point -- exactly as
// tests/pt_math_sphere_test.cpp does for intersectSphere.  A mirror that
// has drifted is worthless, so TEST_CASE("shader mirror is still
// faithful") re-reads BOTH .slang files and pins the literals the
// transcription depends on, including the two call sites in
// PathTrace.slang that are the actual subject of the issue.
//
// FAST MATH IS PART OF THE CONTRACT
//
// Metal compiles these modules with its default (fast) math mode, which
// reassociates float add chains freely -- measured on an M4 Max in #267,
// where a Dekker/two-sum implementation of the same accumulator silently
// returned the naive answer.  That is why the accumulator is integer, and
// why this target is built with -ffast-math (see tests/CMakeLists.txt):
// the host mirror faces the same reassociation the GPU applies.  The
// float-side additions this file introduces are individually immune for
// the reasons stated at their tolerance derivations: `fma(rad, rad, k)`
// and `fma(s, s, fma(2 b0, s, k0))` have addends unrelated to their
// products, so there is no algebraic identity for a compiler to exploit,
// and `rad + |oc|` is a sum of two non-negative quantities that has no
// cancellation to lose in the first place.
//
// Deterministic: every input is a literal or derived from literals.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace {

// The mirror must be OPAQUE to the host optimiser.
//
// These functions model a GPU kernel; the exact reference and the
// double-precision input construction next to them do not.  With
// everything inlined into one loop body, clang -ffast-math fuses across
// that boundary and the float32 rounding under measurement stops
// happening -- measured here as the naive form's error collapsing from
// 0.50 m to 8e-5 m at one altitude and not others, i.e. the defect
// disappearing from a test whose whole job is to pin it.  Which way it
// goes depends on surrounding code, so it is not a stable state to
// leave the file in: the numbers this file reports would be a property
// of the inliner rather than of the arithmetic.
//
// noinline restores the boundary.  It does NOT change the operations
// inside each function -- ptFixedAdd still inlines into the accumulator,
// exactly as it does on the GPU -- it only stops the harness from being
// folded into the thing it is measuring.
#if defined(_MSC_VER)
#  define PT_MIRROR __declspec(noinline)
#else
#  define PT_MIRROR __attribute__((noinline))
#endif

// --- shader mirror: shaders/PathTraceMath.slang ---------------------------
// Line-for-line.  F3 rather than glm::vec3 so no vector library gets a
// chance to reassociate a dot product on our behalf.

struct F3 {
    float x, y, z;
};

inline std::uint32_t asuint(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float asfloat(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline F3 sub(F3 a, F3 b) { return F3{a.x - b.x, a.y - b.y, a.z - b.z}; }

// --- finiteness, under -ffast-math ----------------------------------------
//
// std::isfinite() IS UNUSABLE IN THIS FILE, and so is a plain bit test.
// -ffast-math defines __FINITE_MATH_ONLY__, under which (a) isfinite() is
// a builtin the compiler folds to `true`, and (b) LLVM propagates the
// ninf/nnan flags of the producing FP op through a bitcast, so even
// reading the exponent field of the result folds to "finite" as well.
// Both were measured on Apple clang 17, and that is exactly how PR #273
// shipped an assertion that was vacuous on Mac and red on all three CI
// toolchains: the function really did return +inf, and the check said
// otherwise.
//
// A volatile round-trip breaks the chain -- the load carries no fast-math
// flags, so its result has no known FP class -- and the exponent read is
// then honest.  isFiniteHarnessWorks() below asserts that this is still
// true on whatever compiler is running, so the harness can never go
// quietly vacuous again the way std::isfinite did.
inline bool finiteBits(float v) {
    volatile float t = v;
    float u = t;
    std::uint32_t b;
    std::memcpy(&b, &u, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu;
}

inline F3 mad(F3 a, F3 d, float s) {
    return F3{a.x + d.x * s, a.y + d.y * s, a.z + d.z * s};
}
inline float len(F3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
inline float maxc(F3 a) {
    return std::max(std::max(std::fabs(a.x), std::fabs(a.y)), std::fabs(a.z));
}

constexpr float kPtStableSphereRadius = 16384.0f;

int ptFloatExp(float v) {
    return static_cast<int>((asuint(v) >> 23) & 0xFFu) - 127;
}
float ptPow2(int n) {
    return asfloat(static_cast<std::uint32_t>(n + 127) << 23);
}

struct PtFixedSum {
    int hi;
    int lo;
    float up_hi, up_lo;
    float dn_hi, dn_lo;
};

PtFixedSum ptFixedBegin(int e) {
    int sh = e - 23;
    int sl = e - 47;
    PtFixedSum a{};
    a.hi = 0;
    a.lo = 0;
    a.up_hi = ptPow2(sh);
    a.dn_hi = ptPow2(-sh);
    a.up_lo = ptPow2(sl);
    a.dn_lo = ptPow2(-sl);
    return a;
}

void ptFixedAdd(PtFixedSum& a, float x) {
    int ih  = static_cast<int>(x * a.dn_hi);
    float r = x - static_cast<float>(ih) * a.up_hi;
    a.hi += ih;
    a.lo += static_cast<int>(r * a.dn_lo);
}

void ptFixedAddProduct(PtFixedSum& a, float x, float y, float sgn) {
    float xh = asfloat(asuint(x) & 0xFFFFF000u), xl = x - xh;
    float yh = asfloat(asuint(y) & 0xFFFFF000u), yl = y - yh;
    ptFixedAdd(a, sgn * (xh * yh));
    ptFixedAdd(a, sgn * (xh * yl));
    ptFixedAdd(a, sgn * (xl * yh));
    ptFixedAdd(a, sgn * (xl * yl));
}

float ptFixedTotal(const PtFixedSum& a) {
    return static_cast<float>(a.hi) * a.up_hi
         + static_cast<float>(a.lo) * a.up_lo;
}

PT_MIRROR float ptPowerOfPoint(F3 oc, float rad) {
    float m = std::max(std::max(std::fabs(oc.x), std::fabs(oc.y)),
                       std::max(std::fabs(oc.z), rad));
    PtFixedSum a = ptFixedBegin(2 * ptFloatExp(m) + 1);
    ptFixedAddProduct(a, oc.x, oc.x, 1.0f);
    ptFixedAddProduct(a, oc.y, oc.y, 1.0f);
    ptFixedAddProduct(a, oc.z, oc.z, 1.0f);
    ptFixedAddProduct(a, rad, rad, -1.0f);
    return ptFixedTotal(a);
}

PT_MIRROR float ptDotExact(F3 a, F3 b) {
    float m = maxc(a) * maxc(b);
    if (!(m > 0.0f)) { return 0.0f; }
    PtFixedSum s = ptFixedBegin(ptFloatExp(m) + 1);
    ptFixedAddProduct(s, a.x, b.x, 1.0f);
    ptFixedAddProduct(s, a.y, b.y, 1.0f);
    ptFixedAddProduct(s, a.z, b.z, 1.0f);
    return ptFixedTotal(s);
}

float largestOperand(F3 p, F3 c, float rad) {
    return std::max(maxc(p), std::max(maxc(c), rad));
}

constexpr int kPtAltitudeMaxExp = 60;

bool ptAltitudeStable(F3 p, F3 c, float rad) {
    if (!(rad > kPtStableSphereRadius)) { return false; }
    return ptFloatExp(largestOperand(p, c, rad)) <= kPtAltitudeMaxExp;
}

PT_MIRROR float ptAltitudeNaive(F3 oc, float rad) {
    int e = ptFloatExp(maxc(oc));
    if (e > kPtAltitudeMaxExp && e <= 127) {
        float m = maxc(oc);
        return m * len(F3{oc.x / m, oc.y / m, oc.z / m}) - rad;
    }
    return len(oc) - rad;
}

PT_MIRROR float ptPowerOfPointAt(F3 p, F3 c, float rad) {
    float m = largestOperand(p, c, rad);
    PtFixedSum a = ptFixedBegin(2 * ptFloatExp(m) + 2);
    ptFixedAddProduct(a, p.x, p.x, 1.0f);
    ptFixedAddProduct(a, p.y, p.y, 1.0f);
    ptFixedAddProduct(a, p.z, p.z, 1.0f);
    ptFixedAddProduct(a, p.x, c.x, -2.0f);
    ptFixedAddProduct(a, p.y, c.y, -2.0f);
    ptFixedAddProduct(a, p.z, c.z, -2.0f);
    ptFixedAddProduct(a, c.x, c.x, 1.0f);
    ptFixedAddProduct(a, c.y, c.y, 1.0f);
    ptFixedAddProduct(a, c.z, c.z, 1.0f);
    ptFixedAddProduct(a, rad, rad, -1.0f);
    return ptFixedTotal(a);
}

PT_MIRROR float ptAltitudeFromPower(float k, float rad) {
    float d = rad + std::sqrt(std::max(std::fma(rad, rad, k), 0.0f));
    return (d > 0.0f) ? (k / d) : 0.0f;
}

PT_MIRROR float ptAltitudeAboveSphere(F3 p, F3 c, float rad) {
    if (!ptAltitudeStable(p, c, rad)) { return ptAltitudeNaive(sub(p, c), rad); }
    return ptAltitudeFromPower(ptPowerOfPointAt(p, c, rad), rad);
}

struct PtRayAltitude {
    F3 ro, rd, c;
    float rad;
    float k0;
    float b0;
    bool stable;
};

PT_MIRROR PtRayAltitude ptRayAltitudeBegin(F3 ro, F3 rd, F3 c, float rad) {
    PtRayAltitude a{};
    a.ro = ro; a.rd = rd; a.c = c; a.rad = rad;
    a.stable = ptAltitudeStable(ro, c, rad);
    a.k0 = a.stable ? ptPowerOfPointAt(ro, c, rad) : 0.0f;
    a.b0 = a.stable ? ptDotExact(sub(ro, c), rd) : 0.0f;
    return a;
}

PT_MIRROR float ptRayAltitudeAt(const PtRayAltitude& a, float s) {
    if (!a.stable || ptFloatExp(s) > kPtAltitudeMaxExp) {
        return ptAltitudeNaive(sub(mad(a.ro, a.rd, s), a.c), a.rad);
    }
    float k = std::fma(s, s, std::fma(2.0f * a.b0, s, a.k0));
    return ptAltitudeFromPower(k, a.rad);
}
// --- end shader mirror ----------------------------------------------------

// The pre-#271 bodies, kept so the tests pin the defect they repair rather
// than only asserting that the new numbers look nice.  Both are the
// expression that was in PathTrace.slang, transcribed unchanged.
PT_MIRROR float planetAltitudeNaive(F3 p, F3 c, float rad) {
    float ox = p.x - c.x, oy = p.y - c.y, oz = p.z - c.z;
    return std::sqrt(ox * ox + oy * oy + oz * oz) - rad;
}

// And the literal substitution issue #271 proposed, on a materialised oc.
// It is a real improvement over the naive form and still not a fix; see
// the header and the test case that pins it.
PT_MIRROR float altitudeViaMaterialisedOc(F3 p, F3 c, float rad) {
    F3 oc = sub(p, c);
    return ptPowerOfPoint(oc, rad) / (len(oc) + rad);
}

// --- reference ------------------------------------------------------------
//
// float * float is EXACT in double (24 + 24 = 48 mantissa bits against
// double's 53), so every term of |p|^2 - 2 p.c + |c|^2 - rad^2 is exact
// and only their sum rounds.  Ten additions, each at most half an ULP of
// a running total bounded by 4m^2, put the numerator within
// 10 * 0.5 * ULP(4m^2) of the value the float32 inputs determine -- at
// Earth radius, 0.08 m^2, which is 6e-9 m of altitude once divided by
// |oc| + rad.  Every error asserted in this file is at least fifty times
// that, so the reference is not the limiting term anywhere.
//
// An earlier revision accumulated these terms in __int128 fixed point to
// remove even that rounding.  It was over-engineering, and it did not
// compile on MSVC (`__int128` is unsupported there), which is what
// Copilot caught on this PR.  Note that a compensated float sum -- Kahan,
// two-sum -- is NOT an option in its place: this target is built
// -ffast-math, and #267 measured such schemes being folded away by
// exactly that.  Plain double summation has no identity to fold, and the
// derived bound above is comfortable, so it is what the reference uses.
//
// The expansion is deliberate: like ptPowerOfPointAt, the reference never
// forms p - c, so it cannot inherit the quantisation that is the subject
// of the measurement.  (In double it would not, but keeping the two in
// the same form makes the comparison exact in intent as well as value.)
double exactPower(F3 p, F3 c, float rad) {
    const double px = p.x, py = p.y, pz = p.z;
    const double cx = c.x, cy = c.y, cz = c.z, r = rad;
    return px * px + py * py + pz * pz
         - 2.0 * (px * cx + py * cy + pz * cz)
         + cx * cx + cy * cy + cz * cz
         - r * r;
}

// Exact altitude of the point p, in double.  The denominator needs only
// relative accuracy (an error there scales h, it does not offset it), and
// double delivers 1e-16 of it, so the denominator contributes nothing;
// the numerator's bound is derived above.
double referenceAltitude(F3 p, F3 c, float rad) {
    double ox = double(p.x) - double(c.x);
    double oy = double(p.y) - double(c.y);
    double oz = double(p.z) - double(c.z);
    return exactPower(p, c, rad)
         / (std::sqrt(ox * ox + oy * oy + oz * oz) + double(rad));
}

// Exact altitude of the UNROUNDED ray sample ro + rd*s.  Rounding that
// position into a float3 is an implementation choice, not part of the
// definition, so the reference must not make it -- that is precisely the
// difference the hoisted march exists to exploit.
double referenceRayAltitude(F3 ro, F3 rd, float s, F3 c, float rad) {
    double ox = double(ro.x) + double(rd.x) * double(s) - double(c.x);
    double oy = double(ro.y) + double(rd.y) * double(s) - double(c.y);
    double oz = double(ro.z) + double(rd.z) * double(s) - double(c.z);
    double l2 = ox * ox + oy * oy + oz * oz;
    return (l2 - double(rad) * double(rad))
         / (std::sqrt(l2) + double(rad));
}

// Distance between consecutive float32 values at |v|.  No formulation can
// beat one ULP of what it returns, so every tolerance here is quoted
// against this rather than as an absolute constant.
double ulpOf(double v) {
    v = std::fabs(v);
    if (!(v > 0.0)) return double(std::numeric_limits<float>::denorm_min());
    return std::ldexp(1.0, ptFloatExp(static_cast<float>(v)) - 23);
}

constexpr double kFloatEps = 1.0 / 16777216.0;      // 2^-24

// Budget for ptAltitudeAboveSphere, derived term by term -- nothing here
// is tuned.  h = k / (rad + |oc|), so:
//
//  * k's own resolution.  ptFixedAddProduct truncates the residue below
//    2^(e-47) once per ptFixedAdd, and ptPowerOfPointAt makes ten exact
//    products = forty of them, so the accumulator can be off by at most
//    40 * 2^(e-47) with e = 2*E + 2 (E the exponent of the largest
//    operand).  Rounding that total into a float32 adds half an ULP of k.
//    Both are errors in m^2 and reach the altitude divided by rad + |oc|.
//  * the denominator.  fma(rad, rad, k) rounds once (half an ULP of
//    |oc|^2, halved again by the sqrt) and the add rounds once, so d
//    carries at most ~0.75 eps relatively; h = k/d inherits that
//    relatively, and the divide itself adds another half ULP of h.
//
// A relative error in the denominator cannot offset h, only scale it,
// which is the property that makes the whole formulation work.
double altitudeBudget(F3 p, F3 c, float rad, double h_ref) {
    int E = ptFloatExp(largestOperand(p, c, rad));
    double k_ref  = std::fabs(h_ref) * (2.0 * double(rad) + std::fabs(h_ref));
    double resid  = 40.0 * std::ldexp(1.0, 2 * E + 2 - 47);
    double denom  = 2.0 * double(rad) + h_ref;
    return (resid + 0.5 * ulpOf(k_ref)) / denom
         + 1.75 * kFloatEps * std::fabs(h_ref)
         + 0.5 * ulpOf(h_ref);
}

// Budget for a hoisted march sample.  k(s) = fma(s, s, fma(2 b0, s, k0)):
// the inner fma rounds once on a value bounded by 2 rad (|h0| + s) and the
// outer once on k itself, so dk <= eps (|inner| + |k|) and the altitude
// picks that up divided by 2 rad -- i.e. eps (|h0| + s + |h|).  The
// remaining terms are ptAltitudeFromPower's, as above.
double marchBudget(F3 ro, F3 c, float rad, double h0, double s, double h_ref) {
    int E = ptFloatExp(largestOperand(ro, c, rad));
    double resid = 40.0 * std::ldexp(1.0, 2 * E + 2 - 47);
    return kFloatEps * (std::fabs(h0) + s + std::fabs(h_ref))
         + resid / (2.0 * double(rad))
         + 1.75 * kFloatEps * std::fabs(h_ref)
         + 0.5 * ulpOf(h_ref);
}

// --- geometry -------------------------------------------------------------

// What Engine.cpp actually pushes: float32 of the IUGG mean Earth radius
// R_1 = (2a + b)/3 = 6 371 008.8 m from WGS-84.  Quoted as the rounded
// float because that is the value the shader receives, and the whole
// question here is what float32 can and cannot represent.
constexpr float kEarthRadius = 6371009.0f;

// The engine's spherical frame: centre directly below the world origin so
// the y = 0 ground plane is tangent to the sphere there (Engine.cpp
// ~8560).  IEEE negation is exact, so this is the exact -R the push
// carries.
const F3 kEngineCentre{0.0f, -kEarthRadius, 0.0f};

// A deliberately non-axis-aligned unit direction.  Axis-aligned setups
// hide this class of defect: with two components zero the roundings line
// up and the naive form looks far better than it is (see #267, and the
// first test case below).
constexpr double kUx = 0.4242640687119285;
constexpr double kUy = 0.565685424949238;
constexpr double kUz = 0.7071067811865476;

// A point at `alt` metres altitude, `horiz` metres of ground arc from the
// world origin, in the engine's frame.
F3 engineFramePoint(double alt, double horiz) {
    double r = double(kEarthRadius) + alt;
    double ang = horiz / double(kEarthRadius);
    return F3{static_cast<float>(r * std::sin(ang) * 0.6),
              static_cast<float>(r * std::cos(ang) - double(kEarthRadius)),
              static_cast<float>(r * std::sin(ang) * 0.8)};
}

struct Sweep {
    double naive = 0.0;
    double materialised_oc = 0.0;
    double stable = 0.0;
    double budget_excess = 0.0;      // max (error - its own derived budget)
};

// Worst case over a 100 km ground sweep at one altitude.  A single probe
// would be a coin toss -- the roundings vary with the low bits of p --
// so every claim below is a maximum over 4001 of them.
Sweep sweepAltitude(double alt) {
    Sweep w;
    for (int i = 0; i <= 4000; ++i) {
        F3 p = engineFramePoint(alt, 1.0e5 * double(i) / 4000.0);
        double ref = referenceAltitude(p, kEngineCentre, kEarthRadius);
        double e_new = std::fabs(
            double(ptAltitudeAboveSphere(p, kEngineCentre, kEarthRadius)) - ref);
        w.naive = std::max(w.naive, std::fabs(
            double(planetAltitudeNaive(p, kEngineCentre, kEarthRadius)) - ref));
        w.materialised_oc = std::max(w.materialised_oc, std::fabs(
            double(altitudeViaMaterialisedOc(p, kEngineCentre, kEarthRadius)) - ref));
        w.stable = std::max(w.stable, e_new);
        w.budget_excess = std::max(
            w.budget_excess,
            e_new - altitudeBudget(p, kEngineCentre, kEarthRadius, ref));
    }
    return w;
}

}  // namespace

// ---------------------------------------------------------------------------

// The scale-free acceptance bound: the repaired altitude lands within a
// few ULPs of the float32 it returns.  Quoted this way rather than as a
// ratio against the naive form on purpose -- the naive error is ~eps*R
// whatever altitude is being measured, while ULP(h) grows with h, so the
// RATIO is 4e6 at a metre and only 64 at 100 km.  The ratio shrinking is
// a fact about float32's resolution of h, not about the fix weakening;
// the ULP bound says the real thing at every scale.  The 1e-6 floor is
// for altitudes at or through zero, where ULP(h) collapses but the
// accumulator's own residue does not.
double stableAcceptance(double alt) {
    return std::max(4.0 * ulpOf(alt), 1.0e-6);
}

TEST_CASE("altitude across scales: 1 m, 1 km and 100 km at Earth radius") {
    // The headline claim, at the three altitudes the issue names plus the
    // eye height the camera actually sits at.  Each row is a worst case
    // over 4001 sample points, not a lucky probe.
    for (double alt : {1.0, 1.7, 100.0, 1000.0, 100000.0}) {
        CAPTURE(alt);
        Sweep w = sweepAltitude(alt);
        CAPTURE(w.naive);
        CAPTURE(w.materialised_oc);
        CAPTURE(w.stable);
        // Every probe is inside its own derived budget.
        CHECK(w.budget_excess <= 0.0);
        // ... and the sweep as a whole is at the representation limit.
        CHECK(w.stable <= stableAcceptance(alt));
        // The defect, pinned: the pre-#271 expression loses a decimetre or
        // more at EVERY altitude, because its error is ~eps*R regardless
        // of how small the altitude being measured is.  That is the whole
        // shape of the problem -- it does not get better near the ground,
        // which is where a scale-height exponential cares most.
        CHECK(w.naive >= 0.1);
        CHECK(w.stable * 50.0 < w.naive);
    }
    // Near the ground, where it matters, state it absolutely as well.
    for (double alt : {1.0, 1.7, 10.0}) {
        CAPTURE(alt);
        Sweep w = sweepAltitude(alt);
        CAPTURE(w.stable);
        CHECK(w.stable < 1.0e-5);          // sub-micron, measured ~3e-7
        CHECK(w.naive > 1.0e5 * w.stable); // five orders
    }
}

TEST_CASE("the exact-numerator substitution alone is not the fix") {
    // Issue #271 proposes h = ptPowerOfPoint(oc, R) / (|oc| + R).  The
    // algebra is right and it is a real improvement, but on a MATERIALISED
    // oc it stops at ~0.25 m -- half the defect -- because p.y - c.y is
    // p.y + R, an addition into a magnitude whose ULP is 0.5 m.  The
    // quantisation happens before any of the clever arithmetic runs.
    //
    // This is the case that justifies ptPowerOfPointAt existing at all, so
    // it is pinned rather than left as a comment.
    for (double alt : {1.0, 1.7, 1000.0}) {
        CAPTURE(alt);
        Sweep w = sweepAltitude(alt);
        CAPTURE(w.materialised_oc);
        CAPTURE(w.stable);
        // Better than naive...
        CHECK(w.materialised_oc < w.naive);
        // ... but still a quarter of a metre, i.e. not a fix.  Note it
        // does not improve with altitude either: 0.25 m is half of
        // ULP(R), and that is all it can ever be.
        CHECK(w.materialised_oc > 0.1);
        CHECK(w.materialised_oc < 0.3);
        // The kernel that never forms p - c is at the representation
        // limit instead, which at these altitudes is orders below.
        CHECK(w.stable <= stableAcceptance(alt));
        CHECK(w.stable * 1000.0 < w.materialised_oc);
    }
}

TEST_CASE("altitude is exact where float32 can represent it exactly") {
    // 6371009 and 6371009 + alt are both integers below 2^24 for these
    // altitudes, so with the point placed on the +y axis every input is
    // exact and the true altitude is exactly `alt`.  No reference
    // arithmetic, nothing to argue with.
    //
    // It is also the warning about how this hid: axis-aligned, oc has two
    // zero components, |oc| is exactly oc.y, and the naive form is exact
    // too.  Any test written on axis-aligned points would have pronounced
    // the old code correct.  The sweeps above do the real work.
    for (float alt : {0.5f, 2.0f, 100.0f, 1024.0f}) {
        CAPTURE(alt);
        F3 p{0.0f, alt, 0.0f};
        CHECK(ptAltitudeAboveSphere(p, kEngineCentre, kEarthRadius) == alt);
        CHECK(planetAltitudeNaive(p, kEngineCentre, kEarthRadius) == alt);
    }
}

TEST_CASE("at the surface, below it, and at the centre") {
    // Sign convention is load-bearing: planetAltitude() may return
    // negative, and the transmittance march clamps that to zero itself.
    // Nothing here branches on the sign -- k is simply negative inside the
    // sphere while the denominator stays positive.
    for (double depth : {0.0, -0.5, -50.0, -1000.0, -100000.0}) {
        CAPTURE(depth);
        Sweep w = sweepAltitude(depth);
        CAPTURE(w.naive);
        CAPTURE(w.stable);
        CHECK(w.budget_excess <= 0.0);
        CHECK(w.stable <= stableAcceptance(depth));
        CHECK(w.naive >= 0.1);
        CHECK(w.stable * 50.0 < w.naive);
    }
    // p at the centre: |oc| = 0, so k = -R^2 exactly and the altitude is
    // exactly -R.  This is the one input where the denominator collapses
    // from ~2R to R, and it must not divide by zero.
    {
        float h = ptAltitudeAboveSphere(kEngineCentre, kEngineCentre, kEarthRadius);
        CHECK(finiteBits(h));
        CHECK(h == -kEarthRadius);
    }
}

TEST_CASE("small radii keep the historic expression") {
    // The gate exists so a miniature-planet scene (r_planet_radius is a
    // cvar, and the engine documents sub-Earth values as legal) renders
    // exactly as it did.
    //
    // Two claims, and they are not the same strength.
    //
    // EXACT: below the threshold the accumulated path is not taken.  That
    // is what actually guarantees the shader keeps its bits, because in
    // Slang both sides then inline to the identical source expression
    // `length(p - c) - rad`, which the mirror-faithfulness case pins.
    //
    // APPROXIMATE: the two HOST spellings -- length(oc) here, the three
    // squares written out longhand in the pre-#271 transcription -- may
    // contract into fma differently under -ffast-math, so they can differ
    // by one rounding of |p - c|.  That is an ULP of rad + h and not of
    // h, because rad + h is the magnitude the sqrt actually produced.
    // Asserting bit-identity across two spellings would be asserting a
    // property of the host inliner, not of the shader.
    for (float rad : {1.0f, 100.0f, 1000.0f, 8192.0f, kPtStableSphereRadius}) {
        for (double alt : {0.0, 0.25, 3.0, 400.0, -12.0}) {
            for (double horiz : {0.0, 1.0, 130.0}) {
                CAPTURE(rad);
                CAPTURE(alt);
                CAPTURE(horiz);
                double r = double(rad) + alt;
                double ang = (double(rad) > 0.0) ? horiz / double(rad) : 0.0;
                F3 c{0.0f, -rad, 0.0f};
                F3 p{static_cast<float>(r * std::sin(ang) * 0.6),
                     static_cast<float>(r * std::cos(ang) - double(rad)),
                     static_cast<float>(r * std::sin(ang) * 0.8)};
                // The exact claim: no accumulated path, ever.
                CHECK_FALSE(ptAltitudeStable(p, c, rad));
                {
                    double want = double(planetAltitudeNaive(p, c, rad));
                    CHECK(std::fabs(double(ptAltitudeAboveSphere(p, c, rad)) - want)
                          <= 2.0 * ulpOf(double(rad) + std::fabs(want)));
                }
                // Nothing depends on more than that: r_planet_radius is
                // either 0, which takes the planar branch entirely, or a
                // real body, which is four orders above the gate, so no
                // scene has ever reached this path.
                F3 rd{0.0f, 1.0f, 0.0f};
                PtRayAltitude a = ptRayAltitudeBegin(p, rd, c, rad);
                CHECK_FALSE(a.stable);
                for (float s : {0.0f, 1.5f, 40.0f}) {
                    double want = double(planetAltitudeNaive(
                        mad(p, rd, s), c, rad));
                    CHECK(std::fabs(double(ptRayAltitudeAt(a, s)) - want)
                          <= 2.0 * ulpOf(double(rad) + std::fabs(want)));
                }
            }
        }
    }
    // The first radius above the threshold is where they diverge, so the
    // gate is where the comment says it is.
    {
        const float rad = kPtStableSphereRadius * 2.0f;
        F3 c{0.0f, -rad, 0.0f};
        F3 p = F3{static_cast<float>(kUx * 3.0), static_cast<float>(kUy * 3.0),
                  static_cast<float>(kUz * 3.0)};
        CHECK(asuint(ptAltitudeAboveSphere(p, c, rad))
              != asuint(planetAltitudeNaive(p, c, rad)));
    }
    // Small radii are not a regression either: where the naive form is
    // already good, the accumulated one is at least as good.
    for (float rad : {32768.0f, 262144.0f}) {
        F3 c{0.0f, -rad, 0.0f};
        F3 p{0.3f, 1.0f, 0.7f};
        double ref = referenceAltitude(p, c, rad);
        double e_new = std::fabs(double(ptAltitudeAboveSphere(p, c, rad)) - ref);
        double e_old = std::fabs(double(planetAltitudeNaive(p, c, rad)) - ref);
        CAPTURE(rad);
        CAPTURE(e_new);
        CAPTURE(e_old);
        CHECK(e_new <= e_old);
    }
}

TEST_CASE("ray march: hoisting is required, not merely cheaper") {
    // Two frames, same sweep.  The engine's frame puts the world origin on
    // the surface, so p(s) = ro + rd*s is small and materialising it costs
    // little; a frame whose origin is the planet CENTRE -- which is what
    // planetary P1 moves toward, and what any orbital camera needs -- makes
    // p(s) ~ R, where one float32 ULP is half a metre and evaluating the
    // altitude at a materialised sample throws away everything the
    // accumulator just bought.
    //
    // The hoisted quadratic never materialises p(s), so it is the same
    // accuracy in both.  That, not the two-fma cost, is why it is the
    // formulation in the shader.
    struct Frame { const char* name; bool centre_at_origin; };
    for (int fi = 0; fi < 2; ++fi) {
        const bool centre_at_origin = (fi == 1);
        CAPTURE(centre_at_origin);
        double worst_naive = 0.0, worst_hoisted = 0.0, worst_perstep = 0.0;
        double worst_excess = -1.0e30;
        for (double cam : {1.7, 1000.0, 10000.0, 100000.0}) {
            for (int ei = -90; ei <= 90; ++ei) {
                double e = double(ei) * 3.14159265358979323846 / 180.0;
                F3 c, ro, rd;
                if (centre_at_origin) {
                    double r = double(kEarthRadius) + cam;
                    c = F3{0.0f, 0.0f, 0.0f};
                    ro = F3{static_cast<float>(kUx * r),
                            static_cast<float>(kUy * r),
                            static_cast<float>(kUz * r)};
                    // A horizontal direction orthogonal to up, tilted by e.
                    double ex = -kUz, ez = kUx;
                    double en = std::sqrt(ex * ex + ez * ez);
                    ex /= en; ez /= en;
                    double dx = kUx * std::sin(e) + ex * std::cos(e);
                    double dy = kUy * std::sin(e);
                    double dz = kUz * std::sin(e) + ez * std::cos(e);
                    double dn = std::sqrt(dx * dx + dy * dy + dz * dz);
                    rd = F3{static_cast<float>(dx / dn),
                            static_cast<float>(dy / dn),
                            static_cast<float>(dz / dn)};
                } else {
                    c = kEngineCentre;
                    ro = F3{0.0f, static_cast<float>(cam), 0.0f};
                    double az = 0.3;
                    double dx = std::cos(e) * std::cos(az), dy = std::sin(e),
                           dz = std::cos(e) * std::sin(az);
                    double dn = std::sqrt(dx * dx + dy * dy + dz * dz);
                    rd = F3{static_cast<float>(dx / dn),
                            static_cast<float>(dy / dn),
                            static_cast<float>(dz / dn)};
                }
                const double t = 300000.0;
                PtRayAltitude a = ptRayAltitudeBegin(ro, rd, c, kEarthRadius);
                REQUIRE(a.stable);
                for (int i = 0; i <= 16; ++i) {
                    float s = static_cast<float>(t * double(i) / 16.0);
                    double ref = referenceRayAltitude(ro, rd, s, c, kEarthRadius);
                    // Only the band the optical-depth integral actually
                    // weights: above ~20 km both scale-height exponentials
                    // have collapsed and altitude accuracy stops meaning
                    // anything physical.
                    if (ref < -1000.0 || ref > 20000.0) continue;
                    F3 p = mad(ro, rd, s);
                    double e_hoist = std::fabs(double(ptRayAltitudeAt(a, s)) - ref);
                    worst_naive = std::max(worst_naive, std::fabs(
                        double(planetAltitudeNaive(p, c, kEarthRadius)) - ref));
                    worst_perstep = std::max(worst_perstep, std::fabs(
                        double(ptAltitudeAboveSphere(p, c, kEarthRadius)) - ref));
                    worst_hoisted = std::max(worst_hoisted, e_hoist);
                    worst_excess = std::max(
                        worst_excess,
                        e_hoist - marchBudget(ro, c, kEarthRadius, cam,
                                              double(s), ref));
                }
            }
        }
        CAPTURE(worst_naive);
        CAPTURE(worst_hoisted);
        CAPTURE(worst_perstep);
        CHECK(worst_excess <= 0.0);
        // The defect, pinned in both frames.
        CHECK(worst_naive > 0.2);
        // The repair: centimetre-or-better in both frames, i.e. at least
        // two orders below the expression it replaces.
        CHECK(worst_hoisted < 0.05);
        CHECK(worst_hoisted * 20.0 < worst_naive);
        if (centre_at_origin) {
            // The point of the whole exercise: evaluating the same exact
            // kernel at a MATERIALISED p(s) collapses back to the naive
            // form's error once the frame origin leaves the surface.
            CHECK(worst_perstep > 0.1);
            CHECK(worst_perstep > 10.0 * worst_hoisted);
        } else {
            // In the engine's own frame it is fine -- which is exactly why
            // this cannot be caught by testing the shipping frame alone.
            CHECK(worst_perstep < 0.05);
        }
    }
}

TEST_CASE("the finiteness harness is not vacuous") {
    // Guard the guard.  If a future compiler folds finiteBits() the way it
    // folds std::isfinite(), every "stays finite" assertion below would
    // pass without testing anything -- which is precisely what happened to
    // PR #273 on Apple clang.  Build an infinity by arithmetic that the
    // compiler cannot see through, and require the harness to catch it.
    float big = kEarthRadius;
    for (int i = 0; i < 8; ++i) big = big * big;      // overflows to +inf
    REQUIRE_FALSE(finiteBits(big));
    REQUIRE(finiteBits(1.0f));
    REQUIRE(finiteBits(0.0f));
    // ... and that std::isfinite is the broken one, not our expectations.
    // Not asserted -- it is compiler-dependent by design -- but captured so
    // a failure elsewhere in this file has the context next to it.
    CAPTURE(std::isfinite(big));
}

TEST_CASE("degenerate inputs stay finite") {
    // THE CONTRACT: finite in, finite out.  Every input below is finite and
    // has a representable altitude, so every result must be finite -- there
    // is no input range carve-out here, because this function is called per
    // sample inside a march and one NaN poisons a whole pixel.
    //
    // Zero radius: the spherical branches never call this way (both gate on
    // rad > 0), but the kernel must not produce a NaN if they ever do.
    {
        float h = ptAltitudeAboveSphere(F3{1.0f, 2.0f, 2.0f}, F3{0, 0, 0}, 0.0f);
        CHECK(finiteBits(h));
        CHECK(double(h) == doctest::Approx(3.0));
    }
    // Everything zero: 0/0 is the one input that would produce a NaN, and
    // ptAltitudeFromPower's guard is what stops it.
    {
        CHECK(ptAltitudeFromPower(0.0f, 0.0f) == 0.0f);
        CHECK(finiteBits(ptAltitudeAboveSphere(F3{0, 0, 0}, F3{0, 0, 0}, 0.0f)));
    }
    // Every exponent from a millimetre to the top of float32, in a general
    // orientation.  Axis-aligned probes miss this: with two components
    // zero, fast math rewrites sqrt(x*x) to |x| and the overflow that the
    // naive path would otherwise hit never happens -- the same degeneracy
    // that hid the original cancellation in #267, now hiding an overflow.
    //
    // Pre-fix measurement (PR #273 as first pushed): NaN for every largest-
    // operand exponent in [64, 74] via the accumulated path, and +inf for
    // everything above it via the naive path.
    for (int E = -10; E <= 126; ++E) {
        CAPTURE(E);
        float u = std::ldexp(1.0f, E);
        F3 p{u * 0.6f, u * 0.5f, u * 0.7f};
        for (float rad : {kEarthRadius, 1.0f, 1.0e18f}) {
            CAPTURE(rad);
            float h = ptAltitudeAboveSphere(p, F3{0, 0, 0}, rad);
            CHECK(finiteBits(h));
        }
    }
    // The same sweep with a non-zero centre, so p - c is a real subtraction
    // and the gate sees two large operands rather than one.
    for (int E = 0; E <= 120; ++E) {
        CAPTURE(E);
        float u = std::ldexp(1.0f, E);
        F3 p{u * 0.6f, u * 0.5f, u * 0.7f};
        F3 c{-u * 0.3f, u * 0.9f, -u * 0.2f};
        float h = ptAltitudeAboveSphere(p, c, kEarthRadius);
        CHECK(finiteBits(h));
    }
    // Ray parameters across the whole float32 range, including the 1e30
    // "no limit" sentinel this engine passes as a ray length elsewhere.
    // Pre-fix: NaN from s = 1e20 upward.
    {
        F3 ro{0.0f, 1.7f, 0.0f};
        for (F3 rd : {F3{0.0f, 1.0f, 0.0f},
                      F3{0.6f, 0.5f, 0.62449980f},      // general, unit
                      F3{0.0f, -1.0f, 0.0f}}) {
            PtRayAltitude a = ptRayAltitudeBegin(ro, rd, kEngineCentre, kEarthRadius);
            for (int E = -10; E <= 126; ++E) {
                CAPTURE(E);
                CHECK(finiteBits(ptRayAltitudeAt(a, std::ldexp(1.0f, E))));
            }
            for (float s : {0.0f, 1.0e6f, 1.0e9f, 1.0e20f, 1.0e30f, 1.0e38f}) {
                CAPTURE(s);
                CHECK(finiteBits(ptRayAltitudeAt(a, s)));
            }
        }
    }
    // The ray entry point over BOTH of its free magnitudes.  The sweep
    // above pins the ray parameter with the origin at eye height; a large
    // origin takes ptRayAltitudeBegin down the ungated branch instead, and
    // the composition of the two is what the march actually is.  Safe by
    // construction -- a large origin makes `stable` false, which routes
    // every sample to the naive form the point sweep already covers -- but
    // an untested composition is exactly what produced the CI failure this
    // case exists for, so it is swept rather than argued.
    for (int Eo = -10; Eo <= 126; Eo += 4) {
        CAPTURE(Eo);
        float u = std::ldexp(1.0f, Eo);
        F3 ro{u * 0.6f, u * 0.5f, u * 0.7f};
        for (F3 rd : {F3{0.0f, 1.0f, 0.0f},
                      F3{0.6f, 0.5f, 0.62449980f},
                      F3{0.0f, -1.0f, 0.0f}}) {
            PtRayAltitude a = ptRayAltitudeBegin(ro, rd, kEngineCentre, kEarthRadius);
            for (int Es = -10; Es <= 126; Es += 4) {
                CAPTURE(Es);
                CHECK(finiteBits(ptRayAltitudeAt(a, std::ldexp(1.0f, Es))));
            }
            CHECK(finiteBits(ptRayAltitudeAt(a, 1.0e30f)));
        }
    }
    // Above the gate the answer is not just finite, it is still RIGHT to
    // the precision the naive form can offer -- the scaled branch must not
    // quietly return nonsense.  At |p| = 2^80 the altitude is |p| to
    // within a rounding, since rad is 73 orders smaller.
    {
        float u = std::ldexp(1.0f, 80);
        F3 p{u * 0.6f, u * 0.5f, u * 0.7f};
        double want = std::sqrt(0.6 * 0.6 + 0.5 * 0.5 + 0.7 * 0.7) * std::ldexp(1.0, 80);
        double got  = double(ptAltitudeAboveSphere(p, F3{0, 0, 0}, kEarthRadius));
        CAPTURE(want);
        CAPTURE(got);
        CHECK_FALSE(ptAltitudeStable(p, F3{0, 0, 0}, kEarthRadius));
        CHECK(std::fabs(got - want) <= 4.0 * ulpOf(want));
    }
    // An infinity or a NaN on the way in is NOT in the domain, but it must
    // not get worse: the gate routes it to the naive form, so an infinite
    // input yields an infinity rather than the NaN that inf/inf would give.
    {
        const float inf = asfloat(0x7F800000u);
        float h = ptAltitudeAboveSphere(F3{inf, 0.0f, 0.0f}, F3{0, 0, 0}, kEarthRadius);
        CHECK_FALSE(ptAltitudeStable(F3{inf, 0, 0}, F3{0, 0, 0}, kEarthRadius));
        CHECK_FALSE(finiteBits(h));          // documented, not repaired
        std::uint32_t b = asuint(h);
        CHECK((b & 0x7FFFFFu) == 0u);        // an infinity, not a NaN
    }
}

TEST_CASE("the accumulator recovers the power of the point exactly") {
    // ptPowerOfPointAt's claim is per-probe and absolute: within the
    // forty-truncation residue of the exact value, every time -- while the
    // naive |p - c|^2 - R^2 is off by the ~2R*0.25 m^2 that materialising
    // p - c costs, which is what the altitude then inherits.
    double worst_naive_ratio = 0.0;
    for (double alt : {1.0, 1.7, 1000.0, -50.0}) {
        for (double horiz : {0.0, 3000.0, 47000.0, 99000.0}) {
            CAPTURE(alt);
            CAPTURE(horiz);
            F3 p = engineFramePoint(alt, horiz);
            double want = exactPower(p, kEngineCentre, kEarthRadius);
            double got  = double(ptPowerOfPointAt(p, kEngineCentre, kEarthRadius));
            int E = ptFloatExp(largestOperand(p, kEngineCentre, kEarthRadius));
            double resid = 40.0 * std::ldexp(1.0, 2 * E + 2 - 47);
            CAPTURE(want);
            CAPTURE(got);
            CHECK(std::fabs(got - want) <= resid + 0.5 * ulpOf(want));
            F3 oc = sub(p, kEngineCentre);
            double naive = double(ptPowerOfPoint(oc, kEarthRadius));
            if (std::fabs(got - want) > 0.0) {
                worst_naive_ratio = std::max(
                    worst_naive_ratio,
                    std::fabs(naive - want) / std::fabs(got - want));
            }
        }
    }
    // The materialised-oc accumulator is exact for the oc it is given and
    // still orders of magnitude off the value p and c actually determine.
    CAPTURE(worst_naive_ratio);
    CHECK(worst_naive_ratio > 100.0);
}

TEST_CASE("shader mirror is still faithful") {
    // The transcription above is only worth something while it matches the
    // shaders.  Pin the literals it depends on -- in PathTraceMath.slang
    // for the kernel, and in PathTrace.slang for the two call sites that
    // are the actual subject of issue #271 -- with whitespace stripped so
    // reformatting cannot fail this.
    auto tighten = [](const char* path) {
        std::ifstream f(path);
        CAPTURE(path);
        REQUIRE(f.good());
        std::stringstream ss;
        ss << f.rdbuf();
        std::string src = ss.str();
        std::string tight;
        tight.reserve(src.size());
        for (char ch : src) {
            if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
        }
        return tight;
    };

    const std::string math = tighten(PT_SHADER_MATH_PATH);
    // The two kernels this issue promoted out of module scope.
    CHECK(math.find("publicfloatptPowerOfPoint(float3oc,floatrad)") != std::string::npos);
    CHECK(math.find("publicfloatptDotExact(float3a,float3b)") != std::string::npos);
    // The gate: shared with intersectSphere, one exponent lower because
    // the seed is 2e+2 rather than 2e+1.
    CHECK(math.find("kPtStableSphereRadius=16384.0") != std::string::npos);
    CHECK(math.find("!(rad>kPtStableSphereRadius)") != std::string::npos);
    CHECK(math.find("kPtAltitudeMaxExp=60") != std::string::npos);
    CHECK(math.find("ptFloatExp(m)<=kPtAltitudeMaxExp") != std::string::npos);
    // The overflow-safe naive form and the per-sample s bound.
    CHECK(math.find("e>kPtAltitudeMaxExp&&e<=127") != std::string::npos);
    CHECK(math.find("returnm*length(oc/m)-rad;") != std::string::npos);
    CHECK(math.find("ptFloatExp(s)>kPtAltitudeMaxExp") != std::string::npos);
    // The accumulator seed and the doubled cross terms that force it.
    CHECK(math.find("ptFixedBegin(2*ptFloatExp(m)+2)") != std::string::npos);
    CHECK(math.find("ptFixedAddProduct(a,p.x,c.x,-2.0)") != std::string::npos);
    CHECK(math.find("ptFixedAddProduct(a,p.y,c.y,-2.0)") != std::string::npos);
    CHECK(math.find("ptFixedAddProduct(a,p.z,c.z,-2.0)") != std::string::npos);
    CHECK(math.find("ptFixedAddProduct(a,rad,rad,-1.0)") != std::string::npos);
    // |oc| recovered from k through a single fma, and the cancellation-free
    // denominator.
    CHECK(math.find("floatd=rad+sqrt(max(fma(rad,rad,k),0.0))") != std::string::npos);
    CHECK(math.find("return(d>0.0)?(k/d):0.0") != std::string::npos);
    // The hoisted quadratic, nested so each fma rounds on its own result.
    CHECK(math.find("floatk=fma(s,s,fma(2.0*a.b0,s,a.k0))") != std::string::npos);
    CHECK(math.find("a.b0=a.stable?ptDotExact(ro-c,rd)") != std::string::npos);
    // The historic expressions behind the gate, unchanged.
    CHECK(math.find("returnlength(oc)-rad;") != std::string::npos);
    CHECK(math.find("returnptAltitudeNaive(a.ro+a.rd*s-a.c,a.rad);") != std::string::npos);
    CHECK(math.find("returnptAltitudeNaive(p-c,rad);") != std::string::npos);

    const std::string pt = tighten(PT_SHADER_PATHTRACE_PATH);
    // planetAltitude's spherical branch, and the planar one it must not
    // have disturbed.
    CHECK(pt.find("if(planet_center_radius.w<=0.0)returnp.y;") != std::string::npos);
    CHECK(pt.find("returnptAltitudeAboveSphere(p,planet_center_radius.xyz,"
                  "planet_center_radius.w);") != std::string::npos);
    // The transmittance march: one hoist, three samples, and the clamp to
    // zero that keeps the optical depth from going negative below ground.
    CHECK(pt.find("PtRayAltituderay_alt=ptRayAltitudeBegin(ro,rd,pc,planet_R);")
          != std::string::npos);
    CHECK(pt.find("h_prev=max(ptRayAltitudeAt(ray_alt,0.0),0.0);") != std::string::npos);
    CHECK(pt.find("floath_mid=max(ptRayAltitudeAt(ray_alt,s_mid),0.0);")
          != std::string::npos);
    CHECK(pt.find("floath_right=max(ptRayAltitudeAt(ray_alt,s_right),0.0);")
          != std::string::npos);
    // And that no naive altitude survived at either site.
    CHECK(pt.find("length(p-planet_center_radius.xyz)-planet_center_radius.w")
          == std::string::npos);
}

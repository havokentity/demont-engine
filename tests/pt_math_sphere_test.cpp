// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for the numerically stable ray-sphere intersection
// (issue #254, phase P0 of #253).
//
// WHAT BROKE
//
// `intersectSphere` used to compute the power of the point as
// `dot(oc, oc) - rad * rad` and then take the near root as
// `-b - sqrt(b*b - k)`.  Both are subtractions of two nearly-equal large
// quantities as soon as the ray origin sits near a large sphere's
// surface, and each throws away about eps*rad of altitude.  At the
// Earth's radius one float32 ULP of |oc|^2 is 4.8e6 m^2 while the true
// k = h(2R + h) at eye height is 2.2e7 m^2 -- three significant digits
// left -- so t came back with ~0.4 m of noise.  Fixing only one of the
// two subtractions buys nothing, because they contribute equally.
//
// The identity k = h(2R + h) is not by itself a fix either: recovering
// h as length(oc) - rad is *itself* a cancelling subtraction carrying
// the same eps*rad error, so the substitution only moves the problem.
// The information is present in the inputs (ro, c and rad are exact
// floats); extracting it needs arithmetic wider than float32.
//
// WHY THIS FILE MIRRORS THE SHADER
//
// The fix lives in shaders/PathTraceMath.slang and there is no host
// entry point to call.  This file transcribes it -- the same operations
// in the same order, since the whole point is the order -- following
// the pattern tests/pt_renderer_light_tree_test.cpp uses for
// pickLightFromTree.  A mirror that has drifted is worthless, so
// TEST_CASE("shader mirror is still faithful") re-reads the .slang and
// pins the handful of literals the transcription depends on.
//
// FAST MATH IS PART OF THE CONTRACT
//
// Metal compiles this module with its default (fast) math mode, which
// reassociates float add chains freely.  That is why the accumulator
// below is integer: every float-only compensated-summation scheme
// (Knuth two-sum, Dekker two-product, Kahan) is defined by an
// expression that is algebraically zero, so a compiler allowed to
// reassociate is allowed to fold it to zero -- measured, on an M4 Max,
// as the stable path silently returning the naive answer.  The test
// target therefore compiles with -ffast-math in Release (see
// tests/CMakeLists.txt) so the host mirror is held to the same standard
// the GPU holds the shader to.  If a future compiler learns a new
// reassociation that breaks the scheme, this test goes red.
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
// happening -- so the numbers this file reports would be a property of
// the inliner rather than of the arithmetic.  Measured here (issue #275
// section 3): with intersectSphere and intersectSphereNaive both inlined,
// clang contracted `b*b - k` into an fma in one and not the other, and
// the "small spheres keep byte-identical arithmetic" case went red on a
// change that did not touch the small-sphere path at all.
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
// Everything in this block is a line-for-line transcription.  Slang's
// float3 is modelled by F3 rather than glm::vec3 so that no vector
// library gets a chance to reassociate a dot product on our behalf.

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

// --- finiteness, under -ffast-math ----------------------------------------
//
// std::isfinite() IS UNUSABLE IN THIS FILE, and so is a plain bit test.
// -ffast-math defines __FINITE_MATH_ONLY__, under which (a) isfinite() is
// a builtin the compiler folds to `true`, and (b) LLVM propagates the
// ninf/nnan flags of the producing FP op through a bitcast, so even
// reading the exponent field of the result folds to "finite" as well.
// Both measured on Apple clang 17 (issue #275 section 2).  This file
// shipped three `CHECK(std::isfinite(t))` assertions in #267 that were
// therefore vacuous on the dev machine and real only on CI -- the same
// shape of defect as #252 and #268: a check that reports success without
// exercising what it names.
//
// A volatile round-trip breaks the chain -- the load carries no fast-math
// flags, so its result has no known FP class -- and the exponent read is
// then honest.  TEST_CASE("the finiteness harness is not vacuous") below
// asserts that this is still true on whatever compiler is running, so the
// harness can never go quietly vacuous again the way std::isfinite did.
// Same helper, same guard, as tests/pt_math_altitude_test.cpp.
inline bool finiteBits(float v) {
    volatile float t = v;
    float u = t;
    std::uint32_t b;
    std::memcpy(&b, &u, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu;
}

constexpr float kPtStableSphereRadius = 16384.0f;

// Largest-operand exponent above which the fixed-point accumulators can no
// longer reach a finite float32 RESULT.  Shared with the altitude kernels;
// the derivation lives at its definition in the shader.
constexpr int kPtAccumMaxExp = 60;

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
    int ih   = static_cast<int>(x * a.dn_hi);
    float r  = x - static_cast<float>(ih) * a.up_hi;
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
    float m = std::max(std::max(std::fabs(a.x), std::fabs(a.y)), std::fabs(a.z))
            * std::max(std::max(std::fabs(b.x), std::fabs(b.y)), std::fabs(b.z));
    if (!(m > 0.0f)) { return 0.0f; }
    PtFixedSum s = ptFixedBegin(ptFloatExp(m) + 1);
    ptFixedAddProduct(s, a.x, b.x, 1.0f);
    ptFixedAddProduct(s, a.y, b.y, 1.0f);
    ptFixedAddProduct(s, a.z, b.z, 1.0f);
    return ptFixedTotal(s);
}

PT_MIRROR bool ptIntersectSphereLarge(F3 oc, F3 rd, float rad, float t_min,
                                      float& t) {
    float b = ptDotExact(oc, rd);
    float k = ptPowerOfPoint(oc, rad);
    float h = std::fma(b, b, -k);
    if (h < 0.0f) { t = 0.0f; return false; }
    h = std::sqrt(h);
    float q  = (b >= 0.0f) ? -(b + h) : (h - b);
    float ta = q;
    float tb = (q != 0.0f) ? k / q : 0.0f;
    float t0 = std::min(ta, tb), t1 = std::max(ta, tb);
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

PT_MIRROR bool ptIntersectSphereScaled(F3 oc, F3 rd, float rad, int ext_exp,
                                       float t_min, float& t) {
    int s = std::min(ext_exp, 126);
    float dn = ptPow2(-s), up = ptPow2(s);
    F3 o{oc.x * dn, oc.y * dn, oc.z * dn};
    float r = rad * dn;
    float b = o.x * rd.x + o.y * rd.y + o.z * rd.z;
    float k = (o.x * o.x + o.y * o.y + o.z * o.z) - r * r;
    float h = b * b - k;
    if (h < 0.0f) { t = 0.0f; return false; }
    h = std::sqrt(h);
    float t0 = (-b - h) * up, t1 = (-b + h) * up;
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

PT_MIRROR bool intersectSphere(F3 ro, F3 rd, F3 c, float rad, float t_min,
                               float& t) {
    F3 oc{ro.x - c.x, ro.y - c.y, ro.z - c.z};
    float m = std::max(std::max(std::fabs(oc.x), std::fabs(oc.y)),
                       std::max(std::fabs(oc.z), rad));
    int ext_exp = ptFloatExp(m);
    if (rad > kPtStableSphereRadius && ext_exp <= kPtAccumMaxExp) {
        return ptIntersectSphereLarge(oc, rd, rad, t_min, t);
    }
    if (ext_exp > kPtAccumMaxExp && ext_exp <= 127) {
        return ptIntersectSphereScaled(oc, rd, rad, ext_exp, t_min, t);
    }
    float b = oc.x * rd.x + oc.y * rd.y + oc.z * rd.z;
    float k = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - rad * rad;
    float h = b * b - k;
    if (h < 0.0f) { t = 0.0f; return false; }
    h = std::sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > t_min) ? t0 : t1;
    return t > t_min;
}

// The pre-#254 body, kept so the tests can pin the defect they repair
// rather than just asserting the new numbers look nice.
PT_MIRROR bool intersectSphereNaive(F3 ro, F3 rd, F3 c, float rad, float& t) {
    F3 oc{ro.x - c.x, ro.y - c.y, ro.z - c.z};
    float b = oc.x * rd.x + oc.y * rd.y + oc.z * rd.z;
    float k = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - rad * rad;
    float h = b * b - k;
    if (h < 0.0f) { t = 0.0f; return false; }
    h = std::sqrt(h);
    float t0 = -b - h, t1 = -b + h;
    t = (t0 > 1e-3f) ? t0 : t1;
    return t > 1e-3f;
}
// --- end shader mirror ----------------------------------------------------

// Reference solve in double.  |oc|^2 needs 46 bits and double carries 53,
// so even the naive expression is exact to ~0.01 m^2 here -- some 1e9
// times finer than the float32 errors under test.  The `a` factor keeps
// it honest for the direction vectors that are only unit to within a
// rounding.
void referenceSolve(F3 ro, F3 rd, F3 c, float rad, double& t) {
    double ox = double(ro.x) - double(c.x);
    double oy = double(ro.y) - double(c.y);
    double oz = double(ro.z) - double(c.z);
    double dx = rd.x, dy = rd.y, dz = rd.z;
    double a = dx * dx + dy * dy + dz * dz;
    double b = ox * dx + oy * dy + oz * dz;
    double k = (ox * ox + oy * oy + oz * oz) - double(rad) * double(rad);
    double h = b * b - a * k;
    if (h < 0.0) { t = 0.0; return; }
    h = std::sqrt(h);
    double q  = (b >= 0.0) ? -(b + h) : (h - b);
    double t0 = q / a, t1 = (q != 0.0) ? k / q : 0.0;
    if (t0 > t1) std::swap(t0, t1);
    t = (t0 > 1e-3) ? t0 : t1;
}

// Distance between consecutive float32 values at |v|.  Every tolerance in
// this file is quoted as a multiple of this rather than as an absolute
// constant: no formulation can beat 1 ULP, so an absolute epsilon would
// either be unmeetable at large t or vacuous at small t.
double ulpOf(double v) {
    v = std::fabs(v);
    if (!(v > 0.0)) return double(std::numeric_limits<float>::denorm_min());
    int e = ptFloatExp(static_cast<float>(v));
    return std::ldexp(1.0, e - 23);
}

// Budget for the stable path, in ULPs of the returned t.
//
// Derived, not tuned.  Three roundings stand between the accumulator and
// the answer: k is correctly rounded (<= 0.5 ULP), q = -b +/- sqrt(disc)
// costs the fma, the sqrt and one add (<= 1.5 ULP together), and the
// divide adds 0.5 ULP.  Call it 3 ULP, doubled to 6 for the conversion
// between "ULP of k" and "ULP of t" when k and t sit either side of a
// binade boundary.  Grazing rays are handled separately below.
constexpr double kStableUlps = 6.0;

// A ray near tangency is ill-conditioned as geometry, not as arithmetic:
// t = k/q with q = -b +/- sqrt(disc), so d(t)/d(disc) ~ 1/(2 sqrt(disc))
// diverges as the chord shrinks to a point.  No formulation removes that.
// The amplification is 1 + |b| / sqrt(disc), computed from the case's own
// geometry in double so the tolerance is derived per case rather than
// picked to make a number come out right.
double grazingAmplification(F3 ro, F3 rd, F3 c, float rad) {
    double ox = double(ro.x) - double(c.x);
    double oy = double(ro.y) - double(c.y);
    double oz = double(ro.z) - double(c.z);
    double b  = ox * rd.x + oy * rd.y + oz * rd.z;
    double k  = (ox * ox + oy * oy + oz * oz) - double(rad) * double(rad);
    double disc = b * b - k;
    if (!(disc > 0.0)) return 1.0;
    return 1.0 + std::fabs(b) / std::sqrt(disc);
}

// Tolerance for a case, chosen by which path the gate sends it down.
//
// Above the gate the accumulated path is limited only by float32's own
// resolution of t, so the budget is kStableUlps ULPs times the grazing
// amplification.  Below the gate the historic expression runs unchanged
// and its error is the cancellation this issue is about: ~eps*rad from
// each of the two subtractions, eps = 2^-24, again amplified near
// tangency.  Four of those covers both subtractions with a factor of two
// to spare.  Asserting the stable budget there would be asserting a fix
// that deliberately was not applied.
double toleranceFor(float rad, double t_ref, double amp) {
    if (rad > kPtStableSphereRadius) {
        return kStableUlps * amp * ulpOf(t_ref);
    }
    constexpr double kFloatEps = 1.0 / 16777216.0;      // 2^-24
    return 4.0 * kFloatEps * double(rad) * amp;
}

// --- geometry builders ----------------------------------------------------

// A general, deliberately non-axis-aligned local up.  Axis-aligned setups
// hide the defect: with two components of oc equal to zero the dot
// products stop cancelling and the naive form looks far better than it is.
constexpr double kUx = 0.4242640687119285;
constexpr double kUy = 0.565685424949238;
constexpr double kUz = 0.7071067811865476;

struct Ray {
    F3 ro, rd;
};

// Origin `dist` metres from the world origin along the general up, aimed
// `tilt_deg` away from straight down.  tilt 0 is nadir, 180 is straight
// out; for a sphere centred on the origin the horizon sits just under 90.
//
// Parameterised by absolute distance rather than by altitude so that the
// finiteness sweeps below can walk the whole float32 exponent range, where
// "radius plus altitude" stops being expressible.
Ray rayAtDistance(double dist, double tilt_deg) {
    F3 ro{float(kUx * dist), float(kUy * dist), float(kUz * dist)};
    // Any unit vector orthogonal to up, to tilt within.
    double ex = -kUy, ey = kUx, ez = 0.0;
    double en = std::sqrt(ex * ex + ey * ey + ez * ez);
    ex /= en; ey /= en; ez /= en;
    double a = tilt_deg * 3.14159265358979323846 / 180.0;
    double dx = -kUx * std::cos(a) + ex * std::sin(a);
    double dy = -kUy * std::cos(a) + ey * std::sin(a);
    double dz = -kUz * std::cos(a) + ez * std::sin(a);
    double dn = std::sqrt(dx * dx + dy * dy + dz * dz);
    return Ray{ro, F3{float(dx / dn), float(dy / dn), float(dz / dn)}};
}

// Origin at altitude `alt` above a sphere of radius `rad` centred on the
// world origin, looking `tilt_deg` away from straight down.
Ray rayAtAltitude(float rad, double alt, double tilt_deg) {
    return rayAtDistance(double(rad) + alt, tilt_deg);
}

// Straight down the +z axis from a known altitude.  With rad and
// rad + alt both integers below 2^24 every input is exact and the answer
// is exactly `alt` -- no reference arithmetic, nothing to argue with.
Ray exactNadirRay(float rad, float alt) {
    return Ray{F3{0.0f, 0.0f, rad + alt}, F3{0.0f, 0.0f, -1.0f}};
}

const F3 kOrigin{0.0f, 0.0f, 0.0f};

// Count occurrences, don't just find one.
//
// A `find(...) != npos` pin asserts "at least one site still says this",
// which is not what these pins mean. Measured while writing #256: reverting
// intersectSphere's historic root selection back to the absolute 1e-3 left
// the other two bodies' `t_min` in place, so every existence pin still
// matched and BOTH mirror tests went green against a shader the mirror no
// longer described. Counting is the fix.
std::size_t countOf(const std::string& hay, const char* needle) {
    std::size_t n = 0, pos = 0;
    const std::size_t len = std::strlen(needle);
    while ((pos = hay.find(needle, pos)) != std::string::npos) { ++n; pos += len; }
    return n;
}

constexpr float kEarthRadius = 6371000.0f;   // metres, 1 unit = 1 metre

// The near cut intersectSphere baked in before #256 made it a parameter.
//
// Every case in this file is about the ARITHMETIC of the #254 kernel --
// how much of |oc|^2 - rad^2 survives, which root gets taken, whether the
// accumulated total stays finite -- and none of that depends on where the
// near cut sits. Holding it at the historic value keeps each assertion
// comparing exactly what it compared before #256, so a change to the
// epsilon scheme cannot silently move a #254 tolerance. The near cut's own
// behaviour is tested in tests/pt_math_epsilon_test.cpp.
constexpr float kHistoricTMin = 1e-3f;

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("planetary radius: exact nadir hit is recovered exactly") {
    // 6371000 and 6371000 + alt are both integers under 2^24, so they are
    // exactly representable and the true distance is exactly `alt` -- no
    // reference arithmetic, nothing to argue with.
    //
    // This case is also a warning about how the defect hid for so long.
    // Axis-aligned, two components of oc are zero, so dot(oc, oc) and b*b
    // are the SAME rounding of oc.z*oc.z and cancel in h = b*b - k, which
    // is enough to hide the ~1.2% error k still carries -- pinned below --
    // and the moment the geometry stops being axis-aligned (the next test
    // case) that error surfaces in t.  Any test written on axis-aligned
    // rays would have called the old code correct.
    //
    // "Would have", not "does": that cancellation only survives while the
    // multiply is NOT fused.  Fuse it and b*b is the exact (R+alt)^2
    // rather than the same rounding of it that k carries, half an ULP of
    // R^2 is left behind, and the near root either drifts by ~0.5 m or
    // drops under the 1e-3 epsilon so the function returns the FAR wall of
    // the planet, 12 742 km away.  Both outcomes were measured in this
    // file on unchanged source, either side of an inlining decision
    // (issue #275 section 3, which is why the mirror is PT_MIRROR now).
    // Neither Metal's default math mode nor the generated SPIR-V
    // constrains contraction, so both are things the pre-#254 shader could
    // do on a real device, and the assertion below pins the disjunction
    // rather than whichever arm this compiler happens to pick.
    for (float alt : {0.5f, 2.0f, 100.0f, 1024.0f}) {
        CAPTURE(alt);
        Ray r = exactNadirRay(kEarthRadius, alt);

        float t_new = 0.0f;
        REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t_new));
        // Exact: not "within a millimetre", bit-for-bit the right answer.
        CHECK(t_new == alt);

        // The naive k does not survive the degeneracy at all (pinned
        // below).  The naive t survives it only unfused: either it lands
        // within the ~eps*R the cancellation leaves, or it misses the near
        // root entirely and reports the far wall.  eps = 2^-24, doubled
        // for the two subtractions, doubled again for margin.
        float t_old = 0.0f;
        REQUIRE(intersectSphereNaive(r.ro, r.rd, kOrigin, kEarthRadius, t_old));
        CAPTURE(t_old);
        const double err_old = std::fabs(double(t_old) - double(alt));
        CAPTURE(err_old);
        CHECK((err_old <= 4.0 * double(kEarthRadius) / 16777216.0
               || double(t_old) > 1.0e6));

        F3 oc{0.0f, 0.0f, kEarthRadius + alt};
        double k_want = double(alt) * (2.0 * double(kEarthRadius) + double(alt));
        double k_new  = double(ptPowerOfPoint(oc, kEarthRadius));
        double k_old  = double((oc.x * oc.x + oc.y * oc.y + oc.z * oc.z)
                               - kEarthRadius * kEarthRadius);
        CAPTURE(k_want);
        CAPTURE(k_new);
        CAPTURE(k_old);
        CHECK(std::fabs(k_new - k_want) <= 0.5 * ulpOf(k_want));
        CHECK(std::fabs(k_old - k_want) > 16.0 * ulpOf(k_want));
    }
}

TEST_CASE("planetary radius: sub-millimetre at a known small altitude") {
    // General orientation, so all three components of oc are large and the
    // dot products really do cancel.  `alt` is the altitude asked for; the
    // float32 origin lands within half an ULP of it, and the reference
    // solve uses the quantised origin, so this measures the arithmetic and
    // not the input rounding.
    for (double alt : {0.25, 1.7, 10.0, 1000.0}) {
        CAPTURE(alt);
        Ray r = rayAtAltitude(kEarthRadius, alt, 0.0);
        double t_ref = 0.0;
        referenceSolve(r.ro, r.rd, kOrigin, kEarthRadius, t_ref);
        REQUIRE(t_ref > 0.0);

        float t_new = 0.0f;
        REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t_new));
        double err_new = std::fabs(double(t_new) - t_ref);
        CAPTURE(err_new);
        CHECK(err_new <= kStableUlps * ulpOf(t_ref));
        // The acceptance claim, stated absolutely.  Only meaningful while
        // t is small enough for float32 to resolve a millimetre at all
        // (ULP crosses 1 mm at t = 8192 m); every alt here is far below.
        CHECK(err_new < 1.0e-4);

        float t_old = 0.0f;
        REQUIRE(intersectSphereNaive(r.ro, r.rd, kOrigin, kEarthRadius, t_old));
        double err_old = std::fabs(double(t_old) - t_ref);
        CAPTURE(err_old);
        // Pin the defect: the old expression is off by more than a
        // centimetre on every one of these, i.e. >100x the new bound.
        CHECK(err_old > 1.0e-2);
        CHECK(err_old > 100.0 * err_new);
    }
}

TEST_CASE("planetary radius: origin exactly on the surface") {
    // |oc| == rad, so k is exactly zero and the naive form's rounding
    // decides the sign of the near root.  When it comes out negative the
    // near root is rejected by the 1e-3 epsilon and the function returns
    // the *far* wall of the planet -- a 12 742 km error, i.e. the ray
    // punches straight through the ground.
    Ray r = rayAtAltitude(kEarthRadius, 0.0, 0.0);
    double t_ref = 0.0;
    referenceSolve(r.ro, r.rd, kOrigin, kEarthRadius, t_ref);

    float t_new = 0.0f;
    REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t_new));
    CHECK(std::fabs(double(t_new) - t_ref) <= kStableUlps * ulpOf(t_ref));
    // Whatever the sub-ULP quantisation of the origin, the hit is on this
    // side of the planet, not the far one.
    CHECK(double(t_new) < 1.0);

    float t_old = 0.0f;
    intersectSphereNaive(r.ro, r.rd, kOrigin, kEarthRadius, t_old);
    CHECK(double(t_old) > 1.0e6);   // the far-wall failure, pinned
}

TEST_CASE("planetary radius: grazing incidence") {
    // Tilt 89 degrees from nadir is a hair below the horizon at these
    // altitudes: a long, shallow chord, the case sub-metre terrain at the
    // horizon actually depends on.
    for (double alt : {2.0, 50.0, 1000.0}) {
        for (double tilt : {80.0, 88.0, 89.0}) {
            CAPTURE(alt);
            CAPTURE(tilt);
            Ray r = rayAtAltitude(kEarthRadius, alt, tilt);
            double t_ref = 0.0;
            referenceSolve(r.ro, r.rd, kOrigin, kEarthRadius, t_ref);
            if (!(t_ref > 0.0)) continue;    // below the horizon, no hit

            float t_new = 0.0f;
            REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t_new));
            double amp = grazingAmplification(r.ro, r.rd, kOrigin, kEarthRadius);
            CAPTURE(amp);
            double err = std::fabs(double(t_new) - t_ref);
            CAPTURE(err);
            CHECK(err <= kStableUlps * amp * ulpOf(t_ref));
        }
    }
}

TEST_CASE("planetary radius: origin inside the sphere") {
    // Only one root is positive, so there is no near/far selection to get
    // wrong -- but t is now ~1.27e7 m, where one float32 ULP is a whole
    // metre.  The answer is representation-limited, and the test says so
    // rather than pretending millimetres are on the table.
    for (double depth : {100.0, 1000.0, 1.0e6}) {
        CAPTURE(depth);
        Ray r = rayAtAltitude(kEarthRadius, -depth, 0.0);
        double t_ref = 0.0;
        referenceSolve(r.ro, r.rd, kOrigin, kEarthRadius, t_ref);
        REQUIRE(t_ref > 0.0);

        float t_new = 0.0f;
        REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t_new));
        double err = std::fabs(double(t_new) - t_ref);
        CAPTURE(err);
        CHECK(err <= kStableUlps * ulpOf(t_ref));
        CHECK(err < 1.0);           // sub-ULP of a 12 742 km chord
    }
}

TEST_CASE("scale sweep: 1 m to 6371 km, nadir and grazing") {
    // The gate sends 1 m and 1 km spheres down the untouched historic
    // path and 1e6 m / 6.371e6 m spheres down the accumulated one.  Both
    // must land inside the same ULP budget; that is the whole claim of
    // the gate -- it switches implementation, not accuracy.
    struct Scale { float rad; double alt; };
    const Scale scales[] = {
        {1.0f,          0.01},
        {1.0f,          0.5},
        {1000.0f,       1.0},
        {1000.0f,       50.0},
        {1000000.0f,    1.0},
        {1000000.0f,    5000.0},
        {kEarthRadius,  1.7},
        {kEarthRadius,  8000.0},
    };
    for (const Scale& s : scales) {
        for (double tilt : {0.0, 30.0, 70.0, 85.0}) {
            CAPTURE(s.rad);
            CAPTURE(s.alt);
            CAPTURE(tilt);
            Ray r = rayAtAltitude(s.rad, s.alt, tilt);
            double t_ref = 0.0;
            referenceSolve(r.ro, r.rd, kOrigin, s.rad, t_ref);
            if (!(t_ref > 0.0)) continue;

            float t_new = 0.0f;
            REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, s.rad, kHistoricTMin, t_new));
            double amp = grazingAmplification(r.ro, r.rd, kOrigin, s.rad);
            double err = std::fabs(double(t_new) - t_ref);
            CAPTURE(amp);
            CAPTURE(err);
            CHECK(err <= toleranceFor(s.rad, t_ref, amp));
            // Whichever path was taken, the answer is sub-millimetre
            // wherever float32 can express a millimetre in t at all.
            if (t_ref < 8192.0) CHECK(err < 1.0e-3);
        }
    }
}

TEST_CASE("small spheres take the historic path unchanged") {
    // The gate exists so that the spheres the renderer actually traces --
    // all of them metres across -- keep byte-identical arithmetic and
    // every golden stays put.  Assert that directly: below the threshold
    // the new function and the pre-#254 one agree bit for bit.
    for (float rad : {0.3f, 1.0f, 12.5f, 500.0f, 8192.0f, kPtStableSphereRadius}) {
        for (double alt : {0.001, 0.25, 3.0, 400.0}) {
            for (double tilt : {0.0, 17.0, 61.0, 88.5}) {
                CAPTURE(rad);
                CAPTURE(alt);
                CAPTURE(tilt);
                Ray r = rayAtAltitude(rad, alt, tilt);
                float t_new = 0.0f, t_old = 0.0f;
                bool h_new = intersectSphere(r.ro, r.rd, kOrigin, rad, kHistoricTMin, t_new);
                bool h_old = intersectSphereNaive(r.ro, r.rd, kOrigin, rad, t_old);
                CHECK(h_new == h_old);
                CHECK(asuint(t_new) == asuint(t_old));
            }
        }
    }
    // ... and the first radius above the threshold is where they diverge,
    // so the gate is where the comment says it is.
    Ray r = rayAtAltitude(kPtStableSphereRadius * 2.0f, 1.0, 0.0);
    float t_new = 0.0f, t_old = 0.0f;
    intersectSphere(r.ro, r.rd, kOrigin, kPtStableSphereRadius * 2.0f, kHistoricTMin, t_new);
    intersectSphereNaive(r.ro, r.rd, kOrigin, kPtStableSphereRadius * 2.0f, t_old);
    CHECK(asuint(t_new) != asuint(t_old));
}

TEST_CASE("the accumulator recovers the power of the point exactly") {
    // General orientation: all three components of oc are ~R, so
    // dot(oc, oc) really does round before rad*rad is taken off it, and
    // no fused multiply-add can rescue the expression.  `want` is the
    // double solve, exact to ~0.01 m^2 out of a k of ~1e7.
    //
    // The accumulator's claim is per-probe and absolute: correctly
    // rounded, every time.  The naive expression's failure is stated as a
    // maximum over the sweep instead, because its error is a fixed
    // ~eps*rad^2 while ULP(k) grows with altitude -- so measured in ULPs
    // of k it is ~rad/(2*alt), which is catastrophic at eye height and
    // merely bad a kilometre up.  Pinning the worst case says the real
    // thing without depending on any single probe's rounding luck.
    double worst_naive_ulps = 0.0;
    for (float rad : {kEarthRadius, 1000000.0f, 32768.0f}) {
        for (double alt : {2.0, 1000.0, -512.0}) {
            CAPTURE(rad);
            CAPTURE(alt);
            Ray r = rayAtAltitude(rad, alt, 0.0);
            F3 oc = r.ro;                                  // centre at origin
            double want = double(oc.x) * oc.x + double(oc.y) * oc.y
                        + double(oc.z) * oc.z - double(rad) * double(rad);
            double got   = double(ptPowerOfPoint(oc, rad));
            double naive = double((oc.x * oc.x + oc.y * oc.y + oc.z * oc.z)
                                  - rad * rad);
            CAPTURE(want);
            CAPTURE(got);
            CAPTURE(naive);
            CHECK(std::fabs(got - want) <= 0.5 * ulpOf(want));
            worst_naive_ulps = std::max(worst_naive_ulps,
                                        std::fabs(naive - want) / ulpOf(want));
        }
    }
    // rad/(2*alt) at Earth radius and 2 m is ~1.6e6 ULPs; three orders of
    // margin below that still separates "catastrophic" from "rounding".
    CAPTURE(worst_naive_ulps);
    CHECK(worst_naive_ulps > 1000.0);
}

TEST_CASE("ptDotExact removes the cancellation from a grazing dot product") {
    // A horizon ray at Earth radius sums three ~6.4e6 terms down to a few
    // kilometres.  The plain dot product keeps only what survives that
    // cancellation, ~eps*R = 0.4 m, which lands straight on t.
    Ray r = rayAtAltitude(kEarthRadius, 2.0, 89.5);
    F3 oc = r.ro;   // centre is the world origin
    double want = double(oc.x) * r.rd.x + double(oc.y) * r.rd.y
                + double(oc.z) * r.rd.z;
    double got = double(ptDotExact(oc, r.rd));
    double naive = double(oc.x * r.rd.x + oc.y * r.rd.y + oc.z * r.rd.z);
    CAPTURE(want);
    CAPTURE(got);
    CAPTURE(naive);
    CHECK(std::fabs(got - want) <= 0.5 * ulpOf(want));
    CHECK(std::fabs(got - want) < std::fabs(naive - want));
}

TEST_CASE("degenerate inputs stay finite") {
    // Ray origin at the centre: k = -rad^2, b = 0, one root each side.
    {
        float t = 0.0f;
        REQUIRE(intersectSphere(kOrigin, F3{0, 0, -1}, kOrigin, kEarthRadius, kHistoricTMin, t));
        CHECK(finiteBits(t));
        CHECK(std::fabs(double(t) - double(kEarthRadius)) <= ulpOf(kEarthRadius));
    }
    // Clean miss: aimed away from a planet-sized sphere.
    {
        Ray r = rayAtAltitude(kEarthRadius, 1000.0, 0.0);
        F3 up{-r.rd.x, -r.rd.y, -r.rd.z};   // straight out, away from it
        float t = 0.0f;
        CHECK_FALSE(intersectSphere(r.ro, up, kOrigin, kEarthRadius, kHistoricTMin, t));
        CHECK(finiteBits(t));
    }
    // Tangent-ish ray that misses by a hair still reports a miss, finitely.
    {
        Ray r = rayAtAltitude(kEarthRadius, 1000.0, 89.5);
        float t = 0.0f;
        bool hit = intersectSphere(r.ro, r.rd, kOrigin, kEarthRadius, kHistoricTMin, t);
        CHECK(finiteBits(t));
        if (hit) CHECK(double(t) > 0.0);
    }
}

TEST_CASE("the finiteness harness is not vacuous") {
    // Guard the guard (issue #275 section 2).  If a future compiler folds
    // finiteBits() the way -ffast-math folds std::isfinite(), every "stays
    // finite" assertion in this file would pass without testing anything --
    // which is exactly what the three std::isfinite() calls this file
    // shipped with did on Apple clang.  Build an infinity by arithmetic the
    // compiler cannot see through and require the harness to catch it.
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

TEST_CASE("finite in, finite out over the whole float32 range") {
    // THE CONTRACT (issue #275 section 1), the same one #271 gave the
    // altitude kernels: every finite (ro, rd, c, rad) yields a finite t.
    // There is no range carve-out, because intersectSphere is called per
    // bounce and a NaN t poisons the whole path, not one sample.
    //
    // Pre-fix measurement, general orientation, on the code this replaces:
    // the gate read `ext_exp <= 74`, which bounds the exponent field
    // ptPow2 has to CONSTRUCT and not the range of the accumulated result.
    // |oc|^2 - rad^2 leaves float32 an entire decade of exponents earlier,
    // so exponents in [61, 74] took the accumulated path and overflowed k
    // to an infinity (-inf for an origin inside the sphere, +inf outside),
    // and everything above 74 fell through to a naive expression whose sum
    // of squares overflowed on its own.  The sweeps below walk every one of
    // those exponents; the assertion counts in the failure output are what
    // pinned the bands.
    //
    // General orientation throughout, and that is load-bearing: with two
    // components of oc equal to zero the dot products stop cancelling AND
    // fast math rewrites sqrt(x*x) to |x|, so the overflow never happens --
    // the same degeneracy that hid the original cancellation in #254, here
    // hiding an overflow.

    // (a) Origin exponent swept from a millimetre to the top of float32,
    //     against a metre-sized, an Earth-sized and a 1e18 m sphere.
    for (int E = -10; E <= 126; ++E) {
        CAPTURE(E);
        double dist = std::ldexp(1.0, E);
        for (float rad : {1.0f, kEarthRadius, 1.0e18f}) {
            CAPTURE(rad);
            for (double tilt : {0.0, 45.0, 89.9, 180.0}) {
                CAPTURE(tilt);
                Ray r = rayAtDistance(dist, tilt);
                float t = 0.0f;
                intersectSphere(r.ro, r.rd, kOrigin, rad, kHistoricTMin, t);
                CHECK(finiteBits(t));
            }
        }
    }

    // (b) Radius exponent swept, with the origin deep inside, on the
    //     surface and outside.  Inside is the case that matters: k is then
    //     ~ -rad^2, so the accumulator overflows NEGATIVE, the discriminant
    //     comes out +inf instead of being rejected, and t = k/q is the
    //     inf/inf the issue names.
    for (int E = 15; E <= 126; ++E) {
        CAPTURE(E);
        float rad = std::ldexp(1.0f, E);
        for (double frac : {0.1, 1.0, 1.5}) {
            CAPTURE(frac);
            for (double tilt : {0.0, 45.0, 89.9}) {
                CAPTURE(tilt);
                Ray r = rayAtDistance(double(rad) * frac, tilt);
                float t = 0.0f;
                intersectSphere(r.ro, r.rd, kOrigin, rad, kHistoricTMin, t);
                CHECK(finiteBits(t));
            }
        }
    }

    // (c) A non-zero centre, so oc = ro - c is a real subtraction and the
    //     gate sees two large operands rather than one.
    for (int E = 0; E <= 120; ++E) {
        CAPTURE(E);
        float u = std::ldexp(1.0f, E);
        F3 ro{u * 0.6f, u * 0.5f, u * 0.7f};
        F3 c{-u * 0.3f, u * 0.9f, -u * 0.2f};
        for (double tilt : {0.0, 45.0, 89.9}) {
            CAPTURE(tilt);
            F3 rd = rayAtDistance(1.0, tilt).rd;
            for (float rad : {kEarthRadius, 1.0e18f}) {
                CAPTURE(rad);
                float t = 0.0f;
                intersectSphere(ro, rd, c, rad, kHistoricTMin, t);
                CHECK(finiteBits(t));
            }
        }
    }

    // (d) Above the gate the answer must not merely be finite, it must
    //     still be RIGHT to the precision the historic expression can
    //     offer -- the rescale must not quietly return nonsense.
    //
    //     Tolerance derived, not tuned.  Above the gate the arithmetic IS
    //     the historic expression, so its ~2 eps*rad cancellation error
    //     stands; the rescale adds three roundings (oc/m, rad/m, and the
    //     final t*m), each <= 0.5 eps relative on a t of ~0.5 rad.  Call it
    //     8 eps*rad, which is the same 4x margin toleranceFor() carries
    //     below the gate.
    {
        constexpr double kFloatEps = 1.0 / 16777216.0;      // 2^-24
        for (int E = 62; E <= 100; E += 2) {
            CAPTURE(E);
            float rad = std::ldexp(1.0f, E);
            Ray r = rayAtDistance(double(rad) * 1.5, 0.0);
            double t_ref = 0.0;
            referenceSolve(r.ro, r.rd, kOrigin, rad, t_ref);
            REQUIRE(t_ref > 0.0);
            float t = 0.0f;
            REQUIRE(intersectSphere(r.ro, r.rd, kOrigin, rad, kHistoricTMin, t));
            double err = std::fabs(double(t) - t_ref);
            CAPTURE(t_ref);
            CAPTURE(err);
            CHECK(err <= 8.0 * kFloatEps * double(rad));
        }
    }
}

TEST_CASE("shader mirror is still faithful") {
    // The transcription above is only worth something while it matches
    // shaders/PathTraceMath.slang.  Pin the literals it depends on, with
    // whitespace stripped so reformatting the shader does not fail this.
    std::ifstream f(PT_SHADER_MATH_PATH);
    REQUIRE_MESSAGE(f.good(), "cannot open " PT_SHADER_MATH_PATH);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    std::string tight;
    tight.reserve(src.size());
    for (char ch : src) {
        if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
    }

    // The gate, and the direction of the comparison.  The upper bound is
    // on the RANGE of the accumulated result, not on the exponent field
    // ptPow2 has to construct (issue #275) -- pin the constant it now
    // shares with the altitude kernels, and the rescaled fallback that
    // covers everything above it.
    CHECK(tight.find("kPtStableSphereRadius=16384.0") != std::string::npos);
    CHECK(tight.find("kPtAccumMaxExp=60") != std::string::npos);
    CHECK(tight.find("rad>kPtStableSphereRadius&&ext_exp<=kPtAccumMaxExp")
          != std::string::npos);
    CHECK(tight.find("ext_exp>kPtAccumMaxExp&&ext_exp<=127") != std::string::npos);
    CHECK(tight.find("ints=min(ext_exp,126)") != std::string::npos);
    CHECK(tight.find("floatdn=ptPow2(-s),up=ptPow2(s)") != std::string::npos);
    CHECK(tight.find("floatt0=(-b-h)*up,t1=(-b+h)*up") != std::string::npos);
    // The near cut is a PARAMETER since #256, not a baked 1e-3. Pin the
    // signatures and every root-selection line that reads it, so a mirror
    // still carrying the old constant cannot pass this file -- which is
    // exactly what happened when #256 first changed the shader: none of
    // the literals above moved, so nothing here noticed.
    CHECK(tight.find("publicboolintersectSphere(float3ro,float3rd,float3c,floatrad,"
                     "floatt_min,outfloatt)") != std::string::npos);
    CHECK(tight.find("publicboolintersectPlane(float3ro,float3rd,float3n,floatd,"
                     "floatt_min,outfloatt)") != std::string::npos);
    // Counted, not merely found: THREE bodies select their root against
    // t_min (accumulated, rescaled, historic) and intersectPlane tests it
    // directly, so an existence pin passes with two of the three reverted.
    CHECK(countOf(tight, "t=(t0>t_min)?t0:t1;returnt>t_min;") == 3u);
    // Four functions end on the near cut: the three above plus
    // intersectPlane, which tests it directly rather than selecting.
    CHECK(countOf(tight, "returnt>t_min;}") == 4u);
    // ... and no executable near cut anywhere in the module still reads
    // the absolute constant.
    CHECK(countOf(tight, ">1e-3)?t0:t1") == 0u);
    CHECK(countOf(tight, "returnt>1e-3;") == 0u);
    // The 12-bit split that makes each partial product exact.
    CHECK(tight.find("asuint(x)&0xFFFFF000u") != std::string::npos);
    CHECK(tight.find("asuint(y)&0xFFFFF000u") != std::string::npos);
    // The two accumulator word scales.
    CHECK(tight.find("intsh=e-23") != std::string::npos);
    CHECK(tight.find("intsl=e-47") != std::string::npos);
    // Exponent surgery rather than exp2(), so fast math cannot approximate it.
    CHECK(tight.find("asfloat(uint(n+127)<<23)") != std::string::npos);
    // The single-rounding discriminant and the Vieta root.
    CHECK(tight.find("floath=fma(b,b,-k)") != std::string::npos);
    CHECK(tight.find("floatq=(b>=0.0)?-(b+h):(h-b)") != std::string::npos);
    CHECK(tight.find("floattb=(q!=0.0)?k/q:0.0") != std::string::npos);
    // The historic small-sphere expression, unchanged.
    CHECK(tight.find("floatk=dot(oc,oc)-rad*rad") != std::string::npos);
    CHECK(tight.find("floatt0=-b-h,t1=-b+h") != std::string::npos);
}

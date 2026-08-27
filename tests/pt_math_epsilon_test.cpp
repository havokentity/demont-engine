// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Unit tests for the scale-relative ray epsilons (issue #256, planetary P2).
//
// WHAT BROKE
//
// Every ray epsilon in the tracer was the absolute constant 1e-3 m: a 1 mm
// normal offset on seventeen secondary-ray origins, a 1 mm `TMin` on the
// mesh query, a 1 mm near-root cut baked inside intersectSphere and
// intersectPlane themselves.  One constant is wrong in BOTH directions the
// moment the scene stops being room-sized:
//
//   TOO SMALL AT RANGE.  A hit point is ro + rd*t, so it carries ~2^-24 of
//   its own magnitude.  At t = 6.4e6 m -- the far limb, the case the
//   planetary arc exists for -- that is metres, thousands of times the
//   offset, and every distant surface shadow-acnes.
//
//   TOO LARGE UP CLOSE.  At t = 0.05 m the numerical error is nanometres
//   and a 1 mm offset is nineteen pixel footprints of peter-panning.
//
// The replacement is ptOffsetRay (Wachter-Binder, Ray Tracing Gems ch. 6)
// for the representational half and 0.25 * cone_width for the geometric
// half.  The derivations live at the definitions in the .slang.
//
// WHAT THIS FILE ASSERTS
//
// Two invariants, at three scales spanning seven orders of magnitude:
//
//   NO ACNE.  The offset origin is strictly ABOVE the true surface, by
//   more than the float32 representation error of the hit point it came
//   from.  This is stated on the geometry rather than on an intersector's
//   return value on purpose: acne is the offset origin landing on the
//   wrong side of the surface, and every intersector inherits that.
//
//   NO PETER-PANNING.  The offset never exceeds ONE ray-cone footprint --
//   one pixel -- so the contact shadow cannot visibly detach, because the
//   detachment is smaller than the pixel that would show it.  That is the
//   invariant an absolute constant cannot have at any scale but one.
//
// Both are checked against the old 1e-3 in the same cases, so the file
// pins the defect it repairs rather than only asserting the new numbers
// look nice.
//
// WHY THIS FILE MIRRORS THE SHADER
//
// The kernels live in shaders/PathTraceMath.slang and there is no host
// entry point to call.  This file transcribes them -- same operations, same
// order -- exactly as tests/pt_math_sphere_test.cpp and
// tests/pt_math_altitude_test.cpp do.  TEST_CASE("shader mirror is still
// faithful") re-reads BOTH .slang files and pins the literals the
// transcription depends on, INCLUDING that the seventeen call sites in
// PathTrace.slang really did convert -- a test that pinned only the kernel
// would pass with every call site left at 1e-3.
//
// FAST MATH IS PART OF THE CONTRACT
//
// Metal compiles these modules with its default (fast) math mode, so the
// host mirror is built with -ffast-math (see tests/CMakeLists.txt) for the
// same reason #267's is.  ptOffsetRay is integer arithmetic on the
// exponent+mantissa field and is immune by construction -- which is the
// point of doing it that way rather than adding a scaled float -- but the
// cone terms around it are ordinary float math and are held to the same
// standard as everything else here.
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

// The mirror must be OPAQUE to the host optimiser -- see the long note in
// tests/pt_math_altitude_test.cpp.  With everything inlined into one loop
// body next to the double-precision reference, clang -ffast-math fuses
// across the boundary and the float32 rounding under measurement stops
// happening, so the numbers would describe the inliner rather than the
// kernel.
#if defined(_MSC_VER)
#  define PT_MIRROR __declspec(noinline)
#else
#  define PT_MIRROR __attribute__((noinline))
#endif

// --- shader mirror: shaders/PathTraceMath.slang ---------------------------
// Line for line.  F3 rather than glm::vec3 so no vector library gets a
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
inline std::int32_t asint(float f) {
    std::int32_t i;
    std::memcpy(&i, &f, 4);
    return i;
}
inline float asfloatI(std::int32_t i) {
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

// std::isfinite() IS UNUSABLE IN THIS FILE -- -ffast-math defines
// __FINITE_MATH_ONLY__, under which Apple clang folds it to `true`, and
// folds a plain exponent-bit read to `true` as well.  See #275 section 2;
// the volatile round-trip is the only honest form.
inline bool finiteBits(float v) {
    volatile float t = v;
    float u = t;
    std::uint32_t b;
    std::memcpy(&b, &u, 4);
    return ((b >> 23) & 0xFFu) != 0xFFu;
}

struct PtRayCone {
    float w;
    float spread;
};

PT_MIRROR PtRayCone ptConeMake(float w, float spread) {
    PtRayCone c{};
    c.w = w;
    c.spread = spread;
    return c;
}

PT_MIRROR PtRayCone ptConeAdvance(PtRayCone c, float t) {
    return ptConeMake(c.w + c.spread * std::max(t, 0.0f), c.spread);
}

PT_MIRROR F3 ptOffsetRay(F3 p, F3 n) {
    const float kOrigin     = 1.0f / 32.0f;
    const float kFloatScale = 1.0f / 65536.0f;
    const float kIntScale   = 256.0f;
    const float pc[3] = {p.x, p.y, p.z};
    const float nc[3] = {n.x, n.y, n.z};
    float out[3];
    for (int k = 0; k < 3; ++k) {
        const std::int32_t of_i = static_cast<std::int32_t>(kIntScale * nc[k]);
        const float p_i =
            asfloatI(asint(pc[k]) + (pc[k] < 0.0f ? -of_i : of_i));
        out[k] = (std::fabs(pc[k]) < kOrigin) ? pc[k] + kFloatScale * nc[k]
                                              : p_i;
    }
    return F3{out[0], out[1], out[2]};
}

PT_MIRROR float ptConeEps(float cone_w, float cos_theta) {
    return 0.25f * cone_w / std::max(cos_theta, 0.25f);
}

PT_MIRROR F3 ptRayOrigin(F3 p, F3 n, PtRayCone cone, F3 dir) {
    F3 o = ptOffsetRay(p, n);
    const float c = std::fabs(n.x * dir.x + n.y * dir.y + n.z * dir.z);
    const float e = ptConeEps(cone.w, c);
    return F3{o.x + n.x * e, o.y + n.y * e, o.z + n.z * e};
}

PT_MIRROR float ptRayTMin(F3 ro, PtRayCone cone) {
    const float kIntScaleRel = 256.0f / 16777216.0f;
    const float m = std::max(std::max(std::fabs(ro.x), std::fabs(ro.y)),
                             std::fabs(ro.z));
    return std::max(0.25f * cone.w, kIntScaleRel * m);
}

// intersectPlane's small-scene body, which is what every fixture takes.
// The near cut is the parameter #256 made it.
PT_MIRROR bool intersectPlane(F3 ro, F3 rd, F3 n, float d, float t_min,
                              float& t) {
    const float denom = n.x * rd.x + n.y * rd.y + n.z * rd.z;
    if (std::fabs(denom) < 1e-4f) { t = 0.0f; return false; }
    t = -((ro.x * n.x + ro.y * n.y + ro.z * n.z) + d) / denom;
    return t > t_min;
}
// --- end shader mirror ----------------------------------------------------

// The pre-#256 origin offset, kept so every case can pin the defect it
// repairs rather than only asserting the new numbers look nice.
PT_MIRROR F3 legacyRayOrigin(F3 p, F3 n) {
    const float kLegacyEps = 1e-3f;
    return F3{p.x + n.x * kLegacyEps,
              p.y + n.y * kLegacyEps,
              p.z + n.z * kLegacyEps};
}

// --- geometry -------------------------------------------------------------

// A deliberately non-axis-aligned unit normal.  Axis-aligned probes hide
// this defect the same way they hid #254's: with two components zero the
// integer offset touches one lane, the dot products stop cancelling, and
// the float32 hit point lands far closer to the true surface than it does
// in general.
constexpr double kNx = 0.4242640687119285;
constexpr double kNy = 0.565685424949238;
constexpr double kNz = 0.7071067811865476;

const F3 kN{float(kNx), float(kNy), float(kNz)};

// Signed distance of `p` from the plane (n.p + d = 0), in double.  The
// plane is the exact one; `p` is whatever float32 produced.
double planeSignedDist(F3 p, F3 n, float d) {
    return double(p.x) * double(n.x) + double(p.y) * double(n.y)
         + double(p.z) * double(n.z) + double(d);
}

// One probe: an eye at the frame origin looking at a ground plane whose
// hit lands `dist` metres away, traced exactly as the shader traces it.
//
// P1 (#255) is what makes this the right construction.  Every GPU-facing
// coordinate is packed relative to the camera anchor, so the eye really is
// near the frame origin and the largest coordinate in play is the trace
// distance itself -- which is the quantity being swept here.
struct Probe {
    F3     hit;        // float32 hit point, exactly as the shader forms it
    F3     rd;         // view direction
    float  d;          // plane offset
    float  t;          // float32 hit distance
    double d0;         // signed distance of `hit` from the true plane
    double err;        // |d0|
    float  cone_w;     // ray-cone footprint at the hit, metres
    double cos_view;   // |dot(n, rd)|
};

// `height_px` / `fov_deg` describe the camera whose pixel footprint sets
// the geometric term.  Both fixture-scale (256 px) and monitor-scale
// (1080 px) cameras are swept, because the offset is a footprint and the
// footprint is a property of the camera, not of the scene.
Probe makeProbe(double dist, double cos_view, int height_px, double fov_deg) {
    Probe pr{};
    // Plane offset: the eye (at the origin) sits `h` above the plane, with
    // h = dist * cos_view so the hit lands at `dist`.
    const double h = dist * cos_view;
    pr.d = float(h);
    // A direction with dot(n, rd) = -cos_view, built from n and a tangent
    // so all three components are large.
    double ex = -kNy, ey = kNx, ez = 0.0;
    const double en = std::sqrt(ex * ex + ey * ey + ez * ez);
    ex /= en; ey /= en; ez /= en;
    const double st = std::sqrt(std::max(0.0, 1.0 - cos_view * cos_view));
    double dx = -kNx * cos_view + ex * st;
    double dy = -kNy * cos_view + ey * st;
    double dz = -kNz * cos_view + ez * st;
    const double dn = std::sqrt(dx * dx + dy * dy + dz * dz);
    pr.rd = F3{float(dx / dn), float(dy / dn), float(dz / dn)};

    const float cone_spread =
        float(2.0 * std::tan(fov_deg * 0.5 * 3.14159265358979323846 / 180.0)
              / double(height_px));

    // Trace it the way the shader does: the camera cone has zero width at
    // the eye, so the near cut there is purely representational.
    const F3 ro{0.0f, 0.0f, 0.0f};
    const PtRayCone cam = ptConeMake(0.0f, cone_spread);
    float t = 0.0f;
    const bool hit = intersectPlane(ro, pr.rd, kN, pr.d,
                                    ptRayTMin(ro, cam), t);
    REQUIRE(hit);
    pr.t   = t;
    pr.hit = F3{ro.x + pr.rd.x * t, ro.y + pr.rd.y * t, ro.z + pr.rd.z * t};
    pr.d0  = planeSignedDist(pr.hit, kN, pr.d);
    pr.err = std::fabs(pr.d0);
    pr.cone_w  = cone_spread * t;
    pr.cos_view = cos_view;
    return pr;
}

// The light direction for a shadow ray leaving the surface at `elev_deg`
// above the tangent plane, in the same plane as the view ray.
F3 lightDir(double elev_deg) {
    const double a = elev_deg * 3.14159265358979323846 / 180.0;
    double ex = -kNy, ey = kNx, ez = 0.0;
    const double en = std::sqrt(ex * ex + ey * ey + ez * ez);
    ex /= en; ey /= en; ez /= en;
    const double sa = std::sin(a), ca = std::cos(a);
    return F3{float(kNx * sa + ex * ca),
              float(kNy * sa + ey * ca),
              float(kNz * sa + ez * ca)};
}

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

// The three scales the acceptance criterion names, in metres.
constexpr double kScales[3] = {1.0, 1.0e3, 6.4e6};

std::string tighten(const char* path) {
    std::ifstream f(path);
    REQUIRE_MESSAGE(f.good(), "cannot open shader");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    std::string tight;
    tight.reserve(src.size());
    for (char ch : src) {
        if (!std::isspace(static_cast<unsigned char>(ch))) tight.push_back(ch);
    }
    return tight;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST_CASE("the finiteness harness is not vacuous") {
    // Guard the guard (#275 section 2).  Build an infinity by arithmetic
    // the compiler cannot see through and require the harness to catch it,
    // so no "stays finite" assertion below can go quietly vacuous.
    float big = 6371000.0f;
    for (int i = 0; i < 8; ++i) big = big * big;
    REQUIRE_FALSE(finiteBits(big));
    REQUIRE(finiteBits(1.0f));
    REQUIRE(finiteBits(0.0f));
    CAPTURE(std::isfinite(big));
}

TEST_CASE("ptOffsetRay steps by the coordinate's own ULP, not by a distance") {
    // The whole claim of the representational term: the step is a fixed
    // number of representable floats, so it grows with the coordinate
    // instead of staying a fixed number of metres.  Measure the ratio at
    // seven orders of magnitude and require it CONSTANT -- that is what
    // "scale-relative" means, stated as a measurement.
    double worst_rel_spread = 0.0;
    double first_rel = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double mag = kScales[i];
        CAPTURE(mag);
        const F3 p{float(kNx * mag), float(kNy * mag), float(kNz * mag)};
        const F3 q = ptOffsetRay(p, kN);
        // Displacement along n, in double.
        const double disp = (double(q.x) - double(p.x)) * kNx
                          + (double(q.y) - double(p.y)) * kNy
                          + (double(q.z) - double(p.z)) * kNz;
        CAPTURE(disp);
        CHECK(disp > 0.0);
        const double rel = disp / mag;
        CAPTURE(rel);
        // 256 ULPs of a coordinate is 256 * 2^-24 = 1.526e-5 relative, to
        // within the binade the coordinate happens to sit in (an ULP is
        // constant across a binade, so the ratio walks over a factor of
        // two and no further).
        CHECK(rel > 0.5 * (256.0 / 16777216.0));
        CHECK(rel < 2.0 * (256.0 / 16777216.0));
        if (i == 0) first_rel = rel;
        else worst_rel_spread = std::max(worst_rel_spread,
                                         std::fabs(rel / first_rel - 1.0));
    }
    // ... and the same relative step at 6.4e6 m as at 1 m, to within that
    // binade factor.  The absolute 1e-3 would be 6.4e6 times wrong here.
    CAPTURE(worst_rel_spread);
    CHECK(worst_rel_spread < 1.0);
}

TEST_CASE("no acne at 1 m, 1 km and 6.4e6 m") {
    // THE FIRST ACCEPTANCE CRITERION.  The offset origin must be strictly
    // ABOVE the true surface -- if it is not, the shadow ray starts inside
    // the geometry it just left and the surface shadows itself.
    //
    // Stated on the geometry rather than on an intersector's return value:
    // acne IS the origin landing on the wrong side, and every intersector
    // inherits it.  `err` is what the float32 hit point actually carries,
    // measured per probe in double rather than assumed.
    for (int height_px : {256, 1080}) {
        for (int i = 0; i < 3; ++i) {
            const double dist = kScales[i];
            for (double cos_view : {0.9, 0.5, 0.15}) {
                CAPTURE(height_px);
                CAPTURE(dist);
                CAPTURE(cos_view);
                const Probe pr = makeProbe(dist, cos_view, height_px,
                                           height_px == 256 ? 55.0 : 60.0);
                CAPTURE(pr.err);
                for (double elev : {90.0, 45.0, 10.0}) {
                    CAPTURE(elev);
                    const F3 L = lightDir(elev);
                    const PtRayCone cone = ptConeMake(pr.cone_w, 0.0f);
                    const F3 sro = ptRayOrigin(pr.hit, kN, cone, L);
                    const double above = planeSignedDist(sro, kN, pr.d);
                    CAPTURE(above);
                    CHECK(finiteBits(float(above)));
                    // Above the surface, and by more than the hit point's
                    // own representation error -- so it is above it for a
                    // reason, not by a rounding that could go the other way
                    // on a different probe.
                    CHECK(above > 0.0);
                    CHECK(above > 2.0 * pr.err);

                    // And the intersector agrees: a shadow ray leaving
                    // this origin does not re-hit the plane it left.
                    float t_self = 0.0f;
                    const bool self_hit =
                        intersectPlane(sro, L, kN, pr.d,
                                       ptRayTMin(sro, cone), t_self);
                    CHECK_FALSE(self_hit);
                }
            }
        }
    }
}

TEST_CASE("no peter-panning at 1 m, 1 km and 6.4e6 m") {
    // THE SECOND ACCEPTANCE CRITERION, and the one that makes the first
    // non-trivial: acne alone is cured by ANY large offset.
    //
    // The invariant is that the offset never exceeds ONE ray-cone
    // footprint.  cone_width at the hit IS the pixel footprint in metres,
    // so an offset inside it cannot produce a visible gap -- the gap is
    // smaller than the pixel that would show it, at every scale, with
    // nothing to tune.  That is exactly what an absolute constant cannot
    // do: it is sub-pixel at one distance and one resolution only.
    double worst_ratio = 0.0;
    for (int height_px : {256, 1080}) {
        for (int i = 0; i < 3; ++i) {
            const double dist = kScales[i];
            for (double cos_view : {0.9, 0.5, 0.15}) {
                CAPTURE(height_px);
                CAPTURE(dist);
                CAPTURE(cos_view);
                const Probe pr = makeProbe(dist, cos_view, height_px,
                                           height_px == 256 ? 55.0 : 60.0);
                for (double elev : {90.0, 45.0, 10.0}) {
                    CAPTURE(elev);
                    const F3 L = lightDir(elev);
                    const PtRayCone cone = ptConeMake(pr.cone_w, 0.0f);
                    const F3 sro = ptRayOrigin(pr.hit, kN, cone, L);
                    const double above = planeSignedDist(sro, kN, pr.d);
                    const double ratio = above / double(pr.cone_w);
                    CAPTURE(pr.cone_w);
                    CAPTURE(above);
                    CAPTURE(ratio);
                    // The GEOMETRIC term alone is at most one footprint --
                    // that is the bound ptConeEps's 0.25 floor was derived
                    // from, and it is exact, not empirical.
                    const float cosL = std::fabs(kN.x * L.x + kN.y * L.y
                                                 + kN.z * L.z);
                    CHECK(double(ptConeEps(pr.cone_w, cosL))
                          <= double(pr.cone_w));
                    // The TOTAL adds the representational step on top. An
                    // ULP is at most 2^-23 of its own value, so 256 of them
                    // is at most 256*2^-23 relative -- three orders below
                    // the footprint at every camera this engine renders at,
                    // which is why the composition is a max in everything
                    // but syntax.
                    const double rep_bound =
                        2.0 * (256.0 / 16777216.0) * dist;
                    CAPTURE(rep_bound);
                    CHECK(above <= double(pr.cone_w) + rep_bound);
                    worst_ratio = std::max(worst_ratio, ratio);
                }
            }
        }
    }
    // The bound is not slack: the grazing widening is designed to reach
    // exactly one footprint at its floor and stop, so the worst case
    // should sit near it rather than orders below.  A worst case far below
    // 0.25 would mean the geometric term had stopped doing anything, and a
    // worst case far above 1 would mean the widening had escaped its floor.
    CAPTURE(worst_ratio);
    CHECK(worst_ratio > 0.2);
    CHECK(worst_ratio < 1.05);
}

TEST_CASE("the absolute 1e-3 fails both criteria, in opposite directions") {
    // Pin the defect.  Without this the two cases above are just numbers;
    // with it they are a repair.
    //
    // FAR: at 6.4e6 m the float32 hit point is DECIMETRES off the true
    // surface, so which side of it the hit lands on is decided by a
    // rounding -- and a 1 mm offset is three orders too small to have a
    // say.  Roughly half the pixels therefore start INSIDE the geometry
    // they just left, which is acne, and it is the case the planetary arc
    // exists for.
    //
    // Swept rather than sampled once: a single probe would only report
    // whichever way its own rounding happened to fall, which is exactly
    // the mistake that lets a defect like this hide.
    {
        int legacy_acne = 0, modern_acne = 0, probes = 0, landed_below = 0;
        double worst_err = 0.0;
        for (int i = 0; i < 257; ++i) {
            const double dist = 6.4e6 * (1.0 + double(i) * 1.0e-6);
            const Probe pr = makeProbe(dist, 0.5, 1080, 60.0);
            worst_err = std::max(worst_err, pr.err);
            const double old_above =
                planeSignedDist(legacyRayOrigin(pr.hit, kN), kN, pr.d);
            const PtRayCone cone = ptConeMake(pr.cone_w, 0.0f);
            const double new_above = planeSignedDist(
                ptRayOrigin(pr.hit, kN, cone, lightDir(45.0)), kN, pr.d);
            if (pr.d0   <= 0.0) ++landed_below;
            if (old_above <= 0.0) ++legacy_acne;
            if (new_above <= 0.0) ++modern_acne;
            // Whatever the rounding did, the new offset clears it AND
            // stays inside one footprint. Both, on every probe.
            CHECK(new_above > 2.0 * pr.err);
            CHECK(new_above <= double(pr.cone_w)
                               + 2.0 * (256.0 / 16777216.0) * dist);
            ++probes;
        }
        CAPTURE(probes);
        CAPTURE(worst_err);
        CAPTURE(landed_below);
        CAPTURE(legacy_acne);
        CAPTURE(modern_acne);
        // The representation error alone is two orders past the old offset.
        CHECK(worst_err > 100.0 * 1.0e-3);
        // Some hits land below the true surface -- which is a rounding, not
        // a bug, and is precisely the thing an origin offset exists to
        // absorb.
        CHECK(landed_below > 0);
        // The old constant absorbs NONE of them: every hit that landed
        // below stays below after a 1 mm lift, because 1 mm is two orders
        // under the error. That equality is the defect, stated exactly --
        // not "roughly half the pixels acne", which would depend on how
        // the sweep happened to sample the rounding.
        CHECK(legacy_acne == landed_below);
        // The new scheme absorbs all of them.
        CHECK(modern_acne == 0);
    }
    // NEAR: a contact shot at 5 cm.  The footprint there is tens of
    // microns, so a 1 mm offset is TENS of pixels of detachment -- the
    // shadow visibly leaves its caster.  Already marginal today; obvious
    // the moment the eye can be centimetres from terrain.
    {
        const Probe pr = makeProbe(0.05, 0.9, 1080, 60.0);
        CAPTURE(pr.cone_w);
        const double old_ratio = 1.0e-3 / double(pr.cone_w);
        CAPTURE(old_ratio);
        CHECK(old_ratio > 10.0);          // >10 pixels of peter-panning
        const PtRayCone cone = ptConeMake(pr.cone_w, 0.0f);
        const F3 new_ro = ptRayOrigin(pr.hit, kN, cone, lightDir(45.0));
        const double new_above = planeSignedDist(new_ro, kN, pr.d);
        CAPTURE(new_above);
        CHECK(new_above / double(pr.cone_w) <= 1.0);
    }
    // The two failures are in OPPOSITE directions, which is the whole
    // argument: no single value of the constant fixes both, so it had to
    // stop being a constant.
}

TEST_CASE("the near cut scales too, and never swallows real geometry") {
    // ptRayTMin is the other half: `TMin`, the mesh triangle test and the
    // intersectors' root selection all read it.  Same two terms.
    for (int i = 0; i < 3; ++i) {
        const double dist = kScales[i];
        CAPTURE(dist);
        const Probe pr = makeProbe(dist, 0.5, 1080, 60.0);
        const PtRayCone cone = ptConeMake(pr.cone_w, 0.0f);
        const float tmin = ptRayTMin(pr.hit, cone);
        CAPTURE(pr.cone_w);
        CAPTURE(tmin);
        CHECK(finiteBits(tmin));
        CHECK(tmin > 0.0f);
        // Never more than a quarter of a footprint plus the ULP term, so
        // it cannot hide an occluder the pixel could have resolved.
        CHECK(double(tmin) <= 0.25 * double(pr.cone_w)
                              + 2.0 * (256.0 / 16777216.0) * dist);
        // ... and it grows with the scene rather than staying at 1 mm.
        if (dist >= 1.0e3) CHECK(double(tmin) > 1.0e-3);
    }
    // A camera ray has zero cone width at the eye, so its near cut is
    // purely representational -- and still non-zero, so a ray cannot
    // re-hit the surface it left through pure rounding.
    {
        const F3 ro{1.0f, 2.0f, 3.0f};
        const float tmin = ptRayTMin(ro, ptConeMake(0.0f, 1.0e-3f));
        CAPTURE(tmin);
        CHECK(tmin > 0.0f);
        CHECK(double(tmin) == doctest::Approx(3.0 * 256.0 / 16777216.0));
    }
}

TEST_CASE("the grazing widening is bounded, and the bound is the invariant") {
    // ptConeEps widens as the outgoing ray lies down toward the surface,
    // because a grazing ray stays near the surface far longer.  But an
    // UNBOUNDED 1/cos trades acne for peter-panning one-for-one: the
    // contact shadow displaces by offset * cot(phi), so an offset that
    // grows as 1/cos displaces as 1/cos^2 and the sub-pixel invariant is
    // gone.  The floor is derived from the invariant itself -- requiring
    // 0.25*w/c <= w gives c >= 0.25 exactly -- so the widening runs 1x to
    // 4x and stops.
    const float w = 1.0f;
    CHECK(double(ptConeEps(w, 1.0f))  == doctest::Approx(0.25));
    CHECK(double(ptConeEps(w, 0.5f))  == doctest::Approx(0.5));
    CHECK(double(ptConeEps(w, 0.25f)) == doctest::Approx(1.0));
    // Below the floor it stops widening: still exactly one footprint.
    CHECK(double(ptConeEps(w, 0.05f))  == doctest::Approx(1.0));
    CHECK(double(ptConeEps(w, 0.0f))   == doctest::Approx(1.0));
    CHECK(double(ptConeEps(w, -1.0f))  == doctest::Approx(1.0));
    // Zero footprint (a camera ray) contributes nothing, at any angle.
    for (float c : {1.0f, 0.25f, 0.01f, 0.0f}) {
        CAPTURE(c);
        CHECK(ptConeEps(0.0f, c) == 0.0f);
    }
}

TEST_CASE("the cone advances along the ray") {
    // transmittance() re-anchors the cone at each refraction segment, so
    // the near cut stays correct after a shadow ray has crossed glass and
    // travelled a long way from where it started.
    const PtRayCone c0 = ptConeMake(0.0f, 1.0e-3f);
    const PtRayCone c1 = ptConeAdvance(c0, 1000.0f);
    CHECK(double(c1.w) == doctest::Approx(1.0));
    CHECK(c1.spread == c0.spread);
    const PtRayCone c2 = ptConeAdvance(c1, 1000.0f);
    CHECK(double(c2.w) == doctest::Approx(2.0));
    // A negative or zero advance cannot shrink it.
    CHECK(ptConeAdvance(c1, -5.0f).w == c1.w);
    CHECK(ptConeAdvance(c1, 0.0f).w == c1.w);
}

TEST_CASE("P1's anchoring is what makes Wachter-Binder's constants valid") {
    // NOT decoration.  ptOffsetRay's published constants -- 256 ULPs, and
    // the 1/65536 m floor below 1/32 m -- are calibrated for coordinates
    // of roughly [1e-2, 1e5] m.  Outside that band the integer step is
    // either smaller than the geometry (acne returns) or larger than it
    // (peter-panning returns).
    //
    // P1 (#255) is what bounds the band: every GPU-facing coordinate is
    // packed relative to the camera anchor, and r_origin_rebase_radius
    // bounds how far the anchor may lag the eye.  Pin the default, because
    // if it ever moves by orders of magnitude this offset scheme has to be
    // re-derived rather than merely re-tuned.
    const std::string eng = tighten(PT_ENGINE_CPP_PATH);
    CHECK(eng.find("PT_CVAR(r_origin_rebase_radius,\"65536\"")
          != std::string::npos);
    // The trace distance is the other half of the bound and is NOT capped
    // by the anchor -- a horizon ray is 6.4e6 m long whatever the anchor
    // does -- which is exactly why the geometric term exists and why the
    // acne case above sweeps to 6.4e6 m rather than to 65536 m.
    CHECK(kScales[2] > 65536.0);
}

TEST_CASE("shader mirror is still faithful") {
    // The transcription above is only worth something while it matches the
    // shaders.  Pin the kernels AND the call sites: a test that pinned only
    // the kernel would pass with all seventeen offset sites left at 1e-3,
    // which is the whole of what this issue is about.
    const std::string math = tighten(PT_SHADER_MATH_PATH);

    // --- the kernels ---
    CHECK(math.find("constfloatkOrigin=1.0/32.0;") != std::string::npos);
    CHECK(math.find("constfloatkFloatScale=1.0/65536.0;") != std::string::npos);
    CHECK(math.find("constfloatkIntScale=256.0;") != std::string::npos);
    CHECK(math.find("int3of_i=int3(kIntScale*n);") != std::string::npos);
    CHECK(math.find("float3p_i=asfloat(asint(p)+select(p<float3(0.0),-of_i,of_i));")
          != std::string::npos);
    CHECK(math.find("returnselect(abs(p)<float3(kOrigin),p+kFloatScale*n,p_i);")
          != std::string::npos);
    CHECK(math.find("return0.25*cone_w/max(cos_theta,0.25);") != std::string::npos);
    CHECK(math.find("returnptOffsetRay(p,n)+n*ptConeEps(cone.w,abs(dot(n,dir)));")
          != std::string::npos);
    CHECK(math.find("constfloatkIntScaleRel=256.0/16777216.0;") != std::string::npos);
    CHECK(math.find("returnmax(0.25*cone.w,kIntScaleRel*m);") != std::string::npos);
    CHECK(math.find("returnptConeMake(c.w+c.spread*max(t,0.0),c.spread);")
          != std::string::npos);

    // --- the intersectors take the cut as a parameter ---
    // Counted: three root selections plus intersectPlane's direct test.
    // An existence pin here passed with two of the three reverted.
    CHECK(countOf(math, "floatt_min,outfloatt)") == 4u);
    CHECK(countOf(math, "t=(t0>t_min)?t0:t1;returnt>t_min;") == 3u);
    CHECK(countOf(math, ">1e-3)?t0:t1") == 0u);
    CHECK(countOf(math, "returnt>1e-3;") == 0u);

    // --- the call sites really converted ---
    const std::string pt = tighten(PT_SHADER_PATHTRACE_PATH);
    // Not one `nf * 1e-3`-shaped origin offset survives, in any of its
    // spellings (nf, nf_0, nf_s, nf_foam, eo), either sign.
    for (const char* pat : {"nf*1e-3", "nf_0*1e-3", "nf_s*1e-3",
                            "nf_foam*1e-3", "eo*1e-3"}) {
        CAPTURE(pat);
        CHECK(pt.find(pat) == std::string::npos);
    }
    // ... nor the two baked `TMin`s, nor the software triangle cut.
    CHECK(pt.find("r.TMin=1e-3") == std::string::npos);
    CHECK(pt.find("r.TMin=t_min") != std::string::npos);
    CHECK(pt.find("t_tri>t_min") != std::string::npos);
    // The trace surface carries the cone, and derives the cut from it in
    // ONE place rather than making every caller carry a second parameter.
    CHECK(pt.find("HitInfotraceScene(float3ro,float3rd,floatshutter_t01,PtRayConecone){"
                  "floatt_min=ptRayTMin(ro,cone);") != std::string::npos);
    // The cone that sets the texture mip is the SAME object that sets the
    // epsilons -- the property that makes the offset sub-pixel by
    // construction rather than by coincidence.
    CHECK(pt.find("cone_width+=cone_spread*h.t;") != std::string::npos);
    CHECK(pt.find("PtRayConecone_hit=ptConeMake(cone_width,cone_spread);")
          != std::string::npos);
    // Seventeen converted origins: count them, so a site that silently
    // reverts is caught by arithmetic rather than by reading.
    std::size_t sites = 0, pos = 0;
    while ((pos = pt.find("ptRayOrigin(", pos)) != std::string::npos) {
        ++sites;
        pos += 1;
    }
    CAPTURE(sites);
    CHECK(sites >= 17u);

    // --- ReSTIR's separate copy converted too ---
    const std::string rf = tighten(PT_SHADER_RESTIRFINAL_PATH);
    CHECK(rf.find("importPathTraceMath;") != std::string::npos);
    CHECK(rf.find("float3sro=ptRayOrigin(hit_pt,n_c,cone,wi);") != std::string::npos);
    CHECK(rf.find("desc.TMin=t_min;") != std::string::npos);
    CHECK(rf.find("1e-4f") == std::string::npos);
    CHECK(rf.find("t=(t0>t_min)?t0:t1;") != std::string::npos);
}

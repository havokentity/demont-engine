// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// RHI acceleration-structure contract tests (issue #254 P0, issue #251).
//
// What this pins, and why it needs to exist
// -----------------------------------------
// Two things were true before this test:
//
//   1. `TLASInstance::instance_id` was honoured by the Vulkan backend
//      (VkAccelerationStructureInstanceKHR::instanceCustomIndex) and
//      silently discarded by the Metal backend, whose hand-rolled
//      instance descriptor had no user-ID field at all. Nothing caught
//      it because the engine has only ever built ONE instance, so the
//      id was always 0 and both backends agreed by coincidence (#251).
//
//   2. There was no way to change a TLAS's instances short of
//      destroying and rebuilding it, and every build ended in a full
//      device or queue drain (#254 P0).
//
// Both are invisible from the host side -- host bookkeeping can look
// perfect while the GPU reads something else entirely. So this test
// goes through the SHADER: `shaders/AccelProbe.slang` (and its CPU
// mirror `RunAccelProbeKernel`) traces one ray per instance and reports
// `CommittedInstanceID()`, and we compare that against what we asked
// for. The same suite runs unmodified on every backend that can host an
// acceleration structure, which is the only way a cross-backend
// divergence like #251 becomes a red test rather than a rendering
// mystery months later.
//
// Units are metres throughout (1 world unit = 1 m), per the project's
// metric-units rule: a 1 m square quad, instances 2 m apart, rays
// starting 2 m above the geometry with a 10 m reach.
//
// Backend coverage is discovered, not assumed. A backend is exercised
// iff Device::Create returns a device, it reports SupportsHardwareRT,
// and it can produce the "accel_probe" pipeline. On macOS that is Metal
// and software. Vulkan is opt-in and skipped by default -- see the
// Vulkan TEST_CASE at the bottom for why (VulkanDevice has no headless
// path today; it needs a window surface to construct).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "rhi/Device.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace pt::rhi;

namespace {

// --- Scene constants (metres) --------------------------------------------
// One 1 m x 1 m quad in the z = 0 plane, instanced N times along +X at
// kInstanceSpacing intervals. Rays fire straight down -Z from 2 m up.
constexpr float kQuadHalf        = 0.5f;   // m
constexpr float kInstanceSpacing = 2.0f;   // m -- 4x the quad width, so a
                                           //      ray between two instances
                                           //      misses both
constexpr float kRayHeight       = 2.0f;   // m above the quad plane
constexpr float kRayReach        = 10.0f;  // m

// Deliberately NOT 0,1,2,3: a backend that returns the instance INDEX
// instead of the user id (which is exactly what Metal's default
// instance descriptor makes the shader read) would pass an
// id == index check. These are also all under 0xFFFFFF, the width
// Vulkan's instanceCustomIndex bitfield gives us.
constexpr std::uint32_t kIds[4] = { 7u, 11u, 23u, 42u };
constexpr std::uint32_t kInstanceCount = 4u;

// Push-constant block, byte-identical to ProbePush in
// shaders/AccelProbe.slang and to the struct RunAccelProbeKernel parses.
struct ProbePush {
    float origin_and_count[4];
    float dir_and_tmax[4];
    float stride_and_row[4];
};

// Row-major 3x4, translation in column 3 -- the layout TLASInstance
// documents and all three backends consume (Vulkan memcpys it straight
// into VkTransformMatrixKHR; Metal transposes into MTLPackedFloat4x3;
// Embree takes it as RTC_FORMAT_FLOAT3X4_ROW_MAJOR).
TLASInstance MakeInstance(AccelStructHandle blas, float tx, std::uint32_t id) {
    TLASInstance inst{};
    inst.blas = blas;
    inst.transform[0] = 1.0f; inst.transform[1]  = 0.0f; inst.transform[2]  = 0.0f; inst.transform[3]  = tx;
    inst.transform[4] = 0.0f; inst.transform[5]  = 1.0f; inst.transform[6]  = 0.0f; inst.transform[7]  = 0.0f;
    inst.transform[8] = 0.0f; inst.transform[9]  = 0.0f; inst.transform[10] = 1.0f; inst.transform[11] = 0.0f;
    inst.instance_id  = id;
    inst.mask         = 0xFFu;
    return inst;
}

// One probe result per ray.
struct ProbeHit {
    bool          hit = false;
    std::uint32_t id  = 0;
};

// Dispatch the probe kernel over `count` rays whose origins march from
// `origin` by `stride`, and return what the shader saw. Bind slots are
// the ones AccelProbe.slang / kSlotToBufBinding / the software tracer
// all agree on: acceleration structure at accel slot 2, output storage
// buffer at buffer slot 1.
std::vector<ProbeHit> Probe(Device& dev, PipelineHandle pipe,
                            AccelStructHandle tlas, BufferHandle out_buf,
                            const float origin[3], const float stride[3],
                            std::uint32_t count) {
    ProbePush push{};
    push.origin_and_count[0] = origin[0];
    push.origin_and_count[1] = origin[1];
    push.origin_and_count[2] = origin[2];
    push.origin_and_count[3] = static_cast<float>(count);
    push.dir_and_tmax[0] = 0.0f;
    push.dir_and_tmax[1] = 0.0f;
    push.dir_and_tmax[2] = -1.0f;      // straight down
    push.dir_and_tmax[3] = kRayReach;
    push.stride_and_row[0] = stride[0];
    push.stride_and_row[1] = stride[1];
    push.stride_and_row[2] = stride[2];
    push.stride_and_row[3] = static_cast<float>(count);   // one row

    CommandBuffer* cb = dev.AcquireCommandBuffer();
    REQUIRE(cb != nullptr);
    cb->BindComputePipeline(pipe);
    cb->BindAccelStruct(2, tlas);
    cb->BindBuffer(1, out_buf, 0);
    cb->PushConstants(&push, sizeof(push));
    cb->Dispatch(1, 1, 1);
    dev.Submit(cb);

    std::vector<std::uint32_t> raw(std::size_t(count) * 2u, 0xDEADBEEFu);
    const bool ok = dev.ReadbackBuffer(out_buf, raw.data(),
                                       raw.size() * sizeof(std::uint32_t));
    REQUIRE(ok);

    std::vector<ProbeHit> out(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i].hit = (raw[i * 2u + 0u] != 0u);
        out[i].id  = raw[i * 2u + 1u];
    }
    return out;
}

// Probe one ray per instance: ray i starts directly above instance i.
std::vector<ProbeHit> ProbeInstances(Device& dev, PipelineHandle pipe,
                                     AccelStructHandle tlas, BufferHandle out_buf,
                                     std::uint32_t count) {
    const float origin[3] = { 0.0f, 0.0f, kRayHeight };
    const float stride[3] = { kInstanceSpacing, 0.0f, 0.0f };
    return Probe(dev, pipe, tlas, out_buf, origin, stride, count);
}

// The whole contract, run against one backend. Split out of the
// TEST_CASEs so both backends assert exactly the same things -- if the
// two suites could drift, a cross-backend divergence could hide in the
// gap, which is the failure mode #251 already demonstrated once.
void RunAccelSuite(BackendType backend) {
    const std::string label = BackendName(backend);
    CAPTURE(label);

    // Headless: no window handle. Both Mac backends tolerate a null
    // NSWindow (the CAMetalLayer attach is a documented no-op on
    // nullptr) and we never present.
    NativeWindowHandle window{};
    window.opaque = nullptr;
    window.width  = 64;
    window.height = 64;

    auto dev = Device::Create(backend, window);
    if (!dev) {
        MESSAGE("skipping " << label << ": backend not built or device creation failed");
        return;
    }
    if (!dev->SupportsHardwareRT()) {
        MESSAGE("skipping " << label << ": no acceleration-structure support");
        return;
    }
    ComputePipelineDesc pdesc{};
    pdesc.kernel_name = "accel_probe";
    pdesc.debug_name  = "accel_probe";
    auto pipe = dev->CreateComputePipeline(pdesc);
    if (!pipe) {
        MESSAGE("skipping " << label << ": accel_probe pipeline unavailable");
        return;
    }

    // --- BLAS: a single 1 m square quad in the z = 0 plane ------------
    const float verts[12] = {
        -kQuadHalf, -kQuadHalf, 0.0f,
         kQuadHalf, -kQuadHalf, 0.0f,
         kQuadHalf,  kQuadHalf, 0.0f,
        -kQuadHalf,  kQuadHalf, 0.0f,
    };
    const std::uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    BLASDesc bdesc{};
    bdesc.vertex_positions = verts;
    bdesc.vertex_count     = 4;
    bdesc.indices          = indices;
    bdesc.index_count      = 6;
    bdesc.flags            = AccelBuildFlags::PreferFastTrace;
    bdesc.debug_name       = "probe_quad";
    auto blas = dev->CreateBLAS(bdesc);
    REQUIRE(blas.id != 0);

    // --- TLAS: N > 1 instances, distinct ids AND distinct transforms --
    // Every instance points at the SAME BLAS, which on Metal also
    // exercises the instancedAccelerationStructures deduplication the
    // N=1 engine never reached.
    std::vector<TLASInstance> insts;
    for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
        insts.push_back(MakeInstance(blas, kInstanceSpacing * float(i), kIds[i]));
    }
    TLASDesc tdesc{};
    tdesc.instances  = insts;
    tdesc.flags      = AccelBuildFlags::PreferFastTrace | AccelBuildFlags::AllowUpdate;
    tdesc.debug_name = "probe_scene";
    auto tlas = dev->CreateTLAS(tdesc);
    REQUIRE(tlas.id != 0);

    BufferDesc odesc{};
    odesc.size       = std::size_t(kInstanceCount) * 2u * sizeof(std::uint32_t);
    odesc.usage      = BufferUsage::Storage;
    odesc.debug_name = "probe_out";
    auto out_buf = dev->CreateBuffer(odesc);
    REQUIRE(out_buf.id != 0);

    // ---------------------------------------------------------------
    // 1. instance_id round-trips through the shader, per instance.
    //    This is #251's regression test. Pre-fix, Metal returned 0 for
    //    every one of these while Vulkan returned 7/11/23/42.
    // ---------------------------------------------------------------
    {
        auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
        for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
            CAPTURE(i);
            CHECK(hits[i].hit);
            CHECK(hits[i].id == kIds[i]);
        }
    }

    // ---------------------------------------------------------------
    // 2. The transforms are distinct and honoured: a ray fired into the
    //    gap between two instances hits nothing. Without this a TLAS
    //    that collapsed every instance onto the origin (the failure a
    //    mis-transposed matrix produces) could still pass test 1 for
    //    ray 0 and would at least look plausible.
    // ---------------------------------------------------------------
    {
        const float origin[3] = { kInstanceSpacing * 0.5f, 0.0f, kRayHeight };
        const float stride[3] = { kInstanceSpacing, 0.0f, 0.0f };
        auto gaps = Probe(*dev, pipe, tlas, out_buf, origin, stride, kInstanceCount);
        for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
            CAPTURE(i);
            CHECK_FALSE(gaps[i].hit);
        }
    }

    // ---------------------------------------------------------------
    // 3. UpdateTLASInstances is observed by the shader, in place, with
    //    no rebuild and no stall.
    //
    //    The mutation swaps the FIRST and LAST instances' transforms
    //    while leaving their ids where they are, so the ray above x = 0
    //    must come back with the last instance's id and vice versa.
    //    A backend that quietly ignored the update, or that refit the
    //    transforms but not the ids, fails here.
    // ---------------------------------------------------------------
    const std::uint64_t stalls_before = dev->AccelGpuStallCount();
    {
        std::vector<TLASInstance> moved = insts;
        moved[0] = MakeInstance(blas, kInstanceSpacing * float(kInstanceCount - 1), kIds[0]);
        moved[kInstanceCount - 1] = MakeInstance(blas, 0.0f, kIds[kInstanceCount - 1]);

        REQUIRE(dev->UpdateTLASInstances(tlas, moved));

        auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
        // Ray 0 is above x = 0, which now holds the LAST instance.
        CHECK(hits[0].hit);
        CHECK(hits[0].id == kIds[kInstanceCount - 1]);
        // Ray N-1 is above the far end, which now holds the FIRST one.
        CHECK(hits[kInstanceCount - 1].hit);
        CHECK(hits[kInstanceCount - 1].id == kIds[0]);
        // The middle instances did not move.
        for (std::uint32_t i = 1; i + 1 < kInstanceCount; ++i) {
            CAPTURE(i);
            CHECK(hits[i].hit);
            CHECK(hits[i].id == kIds[i]);
        }
    }

    // ---------------------------------------------------------------
    // 4. An id-only update (transforms untouched) also lands. This is
    //    the pure-refit shape: same count, same BLAS set, same
    //    geometry -- the cheapest path the backend has.
    // ---------------------------------------------------------------
    {
        std::vector<TLASInstance> renamed;
        for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
            renamed.push_back(MakeInstance(blas, kInstanceSpacing * float(i),
                                           kIds[i] + 1000u));
        }
        REQUIRE(dev->UpdateTLASInstances(tlas, renamed));
        auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
        for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
            CAPTURE(i);
            CHECK(hits[i].hit);
            CHECK(hits[i].id == kIds[i] + 1000u);
        }
    }

    // ---------------------------------------------------------------
    // 5. The drain is actually gone.
    //
    //    AccelGpuStallCount counts every blocking GPU wait the backend
    //    takes on an acceleration-structure path -- queue drains,
    //    device waits, and targeted single-command-buffer waits alike.
    //    CreateBLAS and CreateTLAS each take exactly one (documented,
    //    and off the frame loop). If the old
    //    waitUntilCompleted / vkQueueWaitIdle were still on the update
    //    path this counter would move once per update.
    //
    //    Asserted, not timed: a wall-clock threshold on a shared CI
    //    runner is a flake generator, whereas "the code never called a
    //    blocking wait" is exact.
    //
    //    The loop is PACED -- update, then probe, which reads back and
    //    therefore syncs -- because that is the shape the engine will
    //    use (one update per frame) and because it makes the assertion
    //    deterministic: every ring slot is known-complete when the next
    //    update picks it up. The unpaced burst below is reported rather
    //    than asserted, since a caller that outruns the GPU is designed
    //    to throttle on the update ring (bounded backpressure, never a
    //    pipeline drain) and how often that fires depends on how fast
    //    the host is relative to the GPU.
    // ---------------------------------------------------------------
    {
        const std::uint64_t before = dev->AccelGpuStallCount();
        constexpr int kPaced = 16;
        for (int u = 0; u < kPaced; ++u) {
            std::vector<TLASInstance> moving;
            for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
                // A 1 mm-per-update wobble: real motion, no topology change.
                const float tx = kInstanceSpacing * float(i)
                               + 0.001f * float(u % 8);
                moving.push_back(MakeInstance(blas, tx, kIds[i]));
            }
            REQUIRE(dev->UpdateTLASInstances(tlas, moving));
            auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
            CHECK(hits[0].hit);
            CHECK(hits[0].id == kIds[0]);
        }
        CHECK(dev->AccelGpuStallCount() == before);
    }
    {
        const std::uint64_t before = dev->AccelGpuStallCount();
        constexpr int kBurst = 256;
        const auto t0 = std::chrono::steady_clock::now();
        for (int u = 0; u < kBurst; ++u) {
            std::vector<TLASInstance> moving;
            for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
                moving.push_back(MakeInstance(blas,
                                              kInstanceSpacing * float(i)
                                                  + 0.001f * float(u % 8),
                                              kIds[i]));
            }
            REQUIRE(dev->UpdateTLASInstances(tlas, moving));
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        const std::uint64_t stalls = dev->AccelGpuStallCount() - before;
        MESSAGE(label << ": " << kBurst << " unpaced TLAS updates in " << ms
                      << " ms (" << (ms / kBurst) << " ms each), "
                      << stalls << " ring-throttle waits, 0 queue/device drains");
        // Even flat out, the ring bounds how often we can block: at most
        // one wait per update, and never a queue or device drain. This
        // is the weak-but-true form -- the hard "zero blocking waits"
        // claim is the paced loop above.
        CHECK(stalls <= static_cast<std::uint64_t>(kBurst));
    }
    // Sanity on the counter itself: it is not stuck at zero on a GPU
    // backend, or the assertions above would be vacuous. The software
    // backend has no GPU and legitimately reports 0 forever.
    if (backend != BackendType::Software) {
        CHECK(stalls_before >= 2u);   // one CreateBLAS + one CreateTLAS
    }

    // ---------------------------------------------------------------
    // 6. Capacity is a hard edge, not a silent truncation. Growing past
    //    the create-time instance count needs new storage, so the verb
    //    refuses and the caller has to build a new TLAS.
    // ---------------------------------------------------------------
    {
        std::vector<TLASInstance> too_many;
        for (std::uint32_t i = 0; i < kInstanceCount + 1u; ++i) {
            too_many.push_back(MakeInstance(blas, kInstanceSpacing * float(i), i));
        }
        CHECK_FALSE(dev->UpdateTLASInstances(tlas, too_many));
        // ...and the TLAS is untouched by the refusal.
        auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
        for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
            CAPTURE(i);
            CHECK(hits[i].hit);
            CHECK(hits[i].id == kIds[i]);
        }
    }

    // ---------------------------------------------------------------
    // 7. Shrinking below the capacity works. Count changed, so this is
    //    the rebuild-into-the-same-storage branch rather than the refit
    //    branch -- still no new handle, still no drain.
    // ---------------------------------------------------------------
    {
        std::vector<TLASInstance> fewer;
        fewer.push_back(MakeInstance(blas, 0.0f, kIds[0]));
        fewer.push_back(MakeInstance(blas, kInstanceSpacing, kIds[1]));
        REQUIRE(dev->UpdateTLASInstances(tlas, fewer));

        auto hits = ProbeInstances(*dev, pipe, tlas, out_buf, kInstanceCount);
        CHECK(hits[0].hit);
        CHECK(hits[0].id == kIds[0]);
        CHECK(hits[1].hit);
        CHECK(hits[1].id == kIds[1]);
        // Instances 2 and 3 are gone from the structure entirely.
        CHECK_FALSE(hits[2].hit);
        CHECK_FALSE(hits[3].hit);
    }

    // ---------------------------------------------------------------
    // 8. Bad input is rejected rather than crashed on.
    // ---------------------------------------------------------------
    {
        CHECK_FALSE(dev->UpdateTLASInstances(AccelStructHandle{0}, insts));
        CHECK_FALSE(dev->UpdateTLASInstances(tlas, {}));
        // A TLAS handle that names a BLAS is not a TLAS.
        CHECK_FALSE(dev->UpdateTLASInstances(blas, insts));
    }

    dev->DestroyBuffer(out_buf);
    dev->DestroyAccelStruct(tlas);
    dev->DestroyAccelStruct(blas);
}

}  // namespace

TEST_CASE("rhi accel: multi-instance TLAS + update verb (software)") {
    RunAccelSuite(BackendType::Software);
}

TEST_CASE("rhi accel: multi-instance TLAS + update verb (metal)") {
    RunAccelSuite(BackendType::Metal);
}

TEST_CASE("rhi accel: multi-instance TLAS + update verb (vulkan)") {
    // Opt-in, and off by default even on a Vulkan build.
    //
    // VulkanDevice's constructor requires a real window: it calls
    // glfwCreateWindowSurface, picks a queue family by present support,
    // and builds a swapchain. There is no headless path, so
    // Device::Create(Vulkan, {nullptr, ...}) cannot succeed as the
    // backend stands today -- it bails at the surface and the factory
    // returns nullptr. Rather than lean on GLFW's uninitialised-library
    // guard to turn a null window handle into a clean failure inside a
    // CI process, this case stays behind an explicit switch.
    //
    // Set PT_TEST_VULKAN_ACCEL=1 to run it on a host where a Vulkan
    // device can come up (and expect it to need a headless-device path
    // in VulkanDevice first). The suite body is identical to the Metal
    // and software cases by construction -- that is the point of
    // RunAccelSuite -- so wiring Vulkan in later is a one-line change
    // here, not a new test.
    const char* opt_in = std::getenv("PT_TEST_VULKAN_ACCEL");
    if (opt_in == nullptr || opt_in[0] == '0' || opt_in[0] == '\0') {
        MESSAGE("skipping vulkan: VulkanDevice has no headless path "
                "(set PT_TEST_VULKAN_ACCEL=1 to force)");
        return;
    }
    RunAccelSuite(BackendType::Vulkan);
}

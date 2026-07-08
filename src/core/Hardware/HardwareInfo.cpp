// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "HardwareInfo.h"
#include "../Log.h"

#include <fmt/format.h>

#include <cstdint>
#include <thread>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <sys/utsname.h>
#endif

namespace pt::hw {

namespace {

Info g_info;
bool g_populated = false;

#if defined(__APPLE__)

std::string SysctlString(const char* name) {
    std::size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string out(len, '\0');
    if (sysctlbyname(name, out.data(), &len, nullptr, 0) != 0) return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

template <typename T>
T SysctlInt(const char* name, T fallback = 0) {
    T value{};
    std::size_t len = sizeof(value);
    if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) return fallback;
    return value;
}

void DetectCpuFeatures(Info& info) {
    std::vector<const char*> feats;
    if (SysctlInt<int>("hw.optional.AdvSIMD", 0))  feats.push_back("NEON");
    if (SysctlInt<int>("hw.optional.arm.FEAT_FP16", 0)) feats.push_back("FP16");
    if (SysctlInt<int>("hw.optional.arm.FEAT_DotProd", 0)) feats.push_back("DotProd");
    if (SysctlInt<int>("hw.optional.arm.FEAT_I8MM", 0)) feats.push_back("I8MM");
    if (feats.empty()) {
        info.cpu_features = "ARM64";
    } else {
        info.cpu_features.clear();
        for (std::size_t i = 0; i < feats.size(); ++i) {
            if (i) info.cpu_features += ' ';
            info.cpu_features += feats[i];
        }
    }
}

void PopulateApple(Info& info) {
    info.cpu_model = SysctlString("machdep.cpu.brand_string");
    if (info.cpu_model.empty()) info.cpu_model = "unknown";

    info.cpu_pcores = SysctlInt<int>("hw.perflevel0.physicalcpu", 0);
    info.cpu_ecores = SysctlInt<int>("hw.perflevel1.physicalcpu", 0);
    if (info.cpu_pcores == 0) {
        // Fall back for non-Apple-Silicon Macs.
        info.cpu_pcores = SysctlInt<int>("hw.physicalcpu", 1);
    }

    DetectCpuFeatures(info);

    auto bytes = SysctlInt<std::uint64_t>("hw.memsize", 0);
    info.ram_total_mb = bytes / (1024ull * 1024ull);

    // OS string from uname plus a sysctl release.
    utsname uts{};
    if (uname(&uts) == 0) {
        info.os_name = fmt::format("{} {}", uts.sysname, uts.release);
    }
    auto product = SysctlString("kern.osproductversion");
    if (!product.empty()) {
        info.os_name = fmt::format("macOS {}", product);
    }
}

#else
// Non-Apple fallback: no sysctl introspection, but leaving cpu_pcores
// at its default 1 sized JobSystem::Init(0) to a SINGLE worker on
// 16-core Windows/Linux boxes (a supported native-Vulkan platform) --
// every CSG bake and ParallelFor serialized onto one thread.
// hardware_concurrency() counts logical cores (SMT included), which
// slightly over-provisions vs the Apple physical-P-core count, but a
// few extra workers beat a 1/16th-throughput job system.
void PopulateApple(Info& info) {
    const unsigned n = std::thread::hardware_concurrency();
    info.cpu_pcores = (n > 0u) ? static_cast<int>(n) : 1;
    info.cpu_ecores = 0;
    LOG_WARN("HardwareInfo: detailed hardware detection is macOS-only; "
             "using hardware_concurrency() = {} for worker sizing", n);
}
#endif

}  // namespace

void Populate() {
    if (g_populated) return;
    PopulateApple(g_info);
    g_populated = true;
}

const Info& GetInfo() { return g_info; }
Info&       MutableInfo() { return g_info; }

}  // namespace pt::hw

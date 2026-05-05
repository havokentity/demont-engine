#pragma once

#include <cstdint>
#include <string>

namespace pt::hw {

struct Info {
    std::string  cpu_model;
    int          cpu_pcores  = 1;     // performance cores
    int          cpu_ecores  = 0;     // efficiency cores
    std::string  cpu_features;        // human-readable SIMD list
    std::uint64_t ram_total_mb = 0;
    std::string  os_name;             // e.g. "macOS 15.1"
    std::string  gpu_name;            // populated lazily by RHI backend
    std::uint64_t gpu_vram_mb  = 0;
    bool         gpu_unified  = true; // Apple Silicon default
    bool         gpu_hwrt     = false;
};

// One-shot population of the static Info. Idempotent.
void Populate();

const Info& GetInfo();
Info&       MutableInfo();   // for backends that fill GPU fields later

}  // namespace pt::hw

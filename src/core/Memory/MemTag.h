#pragma once

#include <cstdint>

namespace pt {

// Categories for memory accounting. Add new tags here as subsystems come
// online; keep `Count` last.
enum class MemTag : std::uint8_t {
    Misc = 0,    // default for untagged allocations
    Scene,
    MeshAssets,
    Textures,
    GpuBuffers,
    Csg,
    Console,
    JobSystem,
    Render,
    Stl,         // future use: STL containers wrapped with our allocator
    Count,
};

constexpr int kMemTagCount = static_cast<int>(MemTag::Count);

constexpr const char* MemTagName(MemTag t) {
    switch (t) {
        case MemTag::Misc:       return "Misc";
        case MemTag::Scene:      return "Scene";
        case MemTag::MeshAssets: return "MeshAssets";
        case MemTag::Textures:   return "Textures";
        case MemTag::GpuBuffers: return "GpuBuffers";
        case MemTag::Csg:        return "Csg";
        case MemTag::Console:    return "Console";
        case MemTag::JobSystem:  return "JobSystem";
        case MemTag::Render:     return "Render";
        case MemTag::Stl:        return "Stl";
        case MemTag::Count:      return "<count>";
    }
    return "?";
}

}  // namespace pt

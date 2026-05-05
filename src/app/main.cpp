#include "core/Log.h"
#include "core/Memory/MemTag.h"
#include "core/Memory/Memory.h"

#include <string>
#include <vector>

namespace {

void DemoTaggedAllocations() {
    // STL allocations inside this scope inherit the Scene tag.
    pt::mem::TagScope scope(pt::MemTag::Scene);
    std::vector<int> v;
    v.reserve(1024);
    for (int i = 0; i < 1024; ++i) v.push_back(i);
    LOG_INFO("Allocated tagged vector with {} ints (tag=Scene)", v.size());
}

void DemoUntaggedAllocations() {
    // Default tag (Misc) attribution.
    auto* arr = new int[256];
    arr[0] = 42;
    LOG_INFO("Allocated untagged array of 256 ints (tag=Misc, sentinel={})", arr[0]);
    delete[] arr;
}

}  // namespace

int main() {
    pt::mem::Init();

    LOG_INFO("hello tracer");
    LOG_INFO("path tracer P0 build -- toolchain + tagged memory online");

    DemoTaggedAllocations();
    DemoUntaggedAllocations();

    pt::mem::PrintReport();

    pt::mem::Shutdown();
    return 0;
}

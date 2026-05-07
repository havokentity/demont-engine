// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "Device.h"
#include "../core/Log.h"

namespace pt::rhi {

// Each linked backend provides one of these factories.  We dispatch by
// type.  If a backend isn't compiled in, the call returns nullptr.
extern std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle&);
#if defined(PT_HAS_METAL_BACKEND)
extern std::unique_ptr<Device> CreateMetalDevice(const NativeWindowHandle&);
#endif
#if defined(PT_HAS_VULKAN_BACKEND)
extern std::unique_ptr<Device> CreateVulkanDevice(const NativeWindowHandle&);
#endif

std::unique_ptr<Device> Device::Create(BackendType type,
                                       const NativeWindowHandle& w) {
    switch (type) {
        case BackendType::None:
            return nullptr;
        case BackendType::Software:
            return CreateSoftwareDevice(w);
        case BackendType::Metal:
#if defined(PT_HAS_METAL_BACKEND)
            return CreateMetalDevice(w);
#else
            LOG_ERROR("Metal backend not built into this binary");
            return nullptr;
#endif
        case BackendType::Vulkan:
#if defined(PT_HAS_VULKAN_BACKEND)
            return CreateVulkanDevice(w);
#else
            LOG_ERROR("Vulkan backend not built into this binary");
            return nullptr;
#endif
    }
    return nullptr;
}

}  // namespace pt::rhi

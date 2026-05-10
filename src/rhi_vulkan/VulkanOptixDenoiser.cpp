// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#include "VulkanOptixDenoiser.h"

#if defined(PT_ENABLE_OPTIX)

#include "VulkanDevice.h"
#include "ExternalHandles.h"

#include "../core/Log.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_types.h>

namespace pt::rhi::vk {

VulkanOptixDenoiser::VulkanOptixDenoiser(VulkanDevice* device, Kind kind)
    : device_(device), kind_(kind) {
}

VulkanOptixDenoiser::~VulkanOptixDenoiser() {
    DestroyResources();
}

bool VulkanOptixDenoiser::Init() {
    if (init_attempted_) return ready_;
    init_attempted_ = true;

    // Scaffold: query CUDA device count + driver presence so the build
    // and link path are verified end-to-end (CUDA runtime + driver libs
    // resolve, OptiX header path is included), but do not actually
    // create the OptiX context or denoiser yet -- that lands in the
    // next commit alongside the Vulkan-CUDA interop wiring. Always
    // returns false so the engine routes back to off.
    int device_count = 0;
    cudaError_t cuda_rt_err = cudaGetDeviceCount(&device_count);
    if (cuda_rt_err != cudaSuccess) {
        LOG_WARN("VulkanOptixDenoiser: cudaGetDeviceCount failed ({}); "
                 "OptiX denoiser unavailable. CUDA runtime / driver may "
                 "be missing or no compatible GPU is present.",
                 cudaGetErrorString(cuda_rt_err));
        return false;
    }

    CUresult cuda_drv_err = cuInit(0);
    if (cuda_drv_err != CUDA_SUCCESS) {
        LOG_WARN("VulkanOptixDenoiser: cuInit failed (CUresult={}); "
                 "OptiX denoiser unavailable.",
                 static_cast<int>(cuda_drv_err));
        return false;
    }

    LOG_INFO("VulkanOptixDenoiser scaffold: CUDA detected ({} device(s)); "
             "OptiX denoiser pipeline build deferred to the next commit. "
             "Engine will fall back to off until then.",
             device_count);
    return false;
}

void VulkanOptixDenoiser::Encode(VkCommandBuffer /*cb*/,
                                 const Device::DenoiseDesc& /*d*/) {
    if (!ready_) return;
    // No-op until the implementation lands.
}

void VulkanOptixDenoiser::DestroyResources() {
    // Real teardown comes with the implementation. Order will be:
    // 1. cuda stream sync
    // 2. free OptiX denoiser scratch + state buffers
    // 3. optixDenoiserDestroy
    // 4. close imported external memory + semaphore handles
    // 5. cuCtxDestroy
    optix_ctx_      = nullptr;
    optix_denoiser_ = nullptr;
    ready_          = false;
}

// ---- ExternalHandles.h helper stubs --------------------------------------
// Real implementations land alongside the interop wiring. Return
// VK_ERROR_FEATURE_NOT_PRESENT so any caller that accidentally hits
// the scaffold path gets a clear error rather than silent garbage.

namespace external {

VkResult ExportMemoryHandle(VkDevice          /*device*/,
                            VkDeviceMemory    /*memory*/,
                            NativeHandle*     /*out*/) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult ExportSemaphoreHandle(VkDevice       /*device*/,
                               VkSemaphore    /*semaphore*/,
                               NativeHandle*  /*out*/) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

void CloseNativeHandle(NativeHandle& h) {
#if defined(_WIN32)
    h.value = nullptr;
#else
    h.value = -1;
#endif
}

}  // namespace external

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_OPTIX

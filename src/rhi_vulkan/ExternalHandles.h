// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Platform abstraction for Vulkan <-> CUDA external memory/semaphore interop.
//
// CUDA-Vulkan interop ships per-OS handle shapes:
//   Windows: NT HANDLE      (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
//                            cudaExternalMemoryHandleTypeOpaqueWin32)
//   Linux:   int file desc  (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
//                            cudaExternalMemoryHandleTypeOpaqueFd)
//
// VulkanOptixDenoiser.cpp uses the constants below + the helper
// signatures (implemented in the .cpp) so that the rest of the
// implementation is platform-agnostic. To add a new platform later
// (FreeBSD, etc.) only this header + the helpers in the .cpp need
// touching.
//
// Linux note: code path is structured but untested as of v0.3.1 -- the
// project is targeted at Windows + NVIDIA RTX today, with a Linux
// validation pass planned once a Linux box is available. See
// docs/LINUX_OPTIX.md when it lands. macOS has no NVIDIA driver path
// and uses MetalFX instead, so this header is meaningfully Win+Linux only.

#pragma once

#if defined(PT_ENABLE_OPTIX)

#if defined(_WIN32)
    // <windows.h> must precede <vulkan/vulkan_win32.h> -- the latter
    // references HANDLE / HINSTANCE / HWND / HMONITOR without declaring
    // them itself. Project-wide defines (NOMINMAX, WIN32_LEAN_AND_MEAN)
    // come from CMake via target_compile_definitions, so this stays minimal.
    #include <windows.h>
#endif

#include <vulkan/vulkan.h>

#if defined(_WIN32)
    #include <vulkan/vulkan_win32.h>
#endif

#include <cuda.h>           // CUDA driver API (CUcontext, CUstream, ...)
#include <cuda_runtime.h>   // CUDA runtime API + interop enums

namespace pt::rhi::vk::external {

// Handle-type constants picked at compile time per platform. Both
// platforms use "opaque" handle types -- the OS-level representation
// differs but the underlying Vulkan resource is identical.
struct HandleTypes {
#if defined(_WIN32)
    static constexpr VkExternalMemoryHandleTypeFlagBits     vk_memory =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    static constexpr VkExternalSemaphoreHandleTypeFlagBits  vk_semaphore =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    static constexpr cudaExternalMemoryHandleType           cuda_memory =
        cudaExternalMemoryHandleTypeOpaqueWin32;
    static constexpr cudaExternalSemaphoreHandleType        cuda_semaphore =
        cudaExternalSemaphoreHandleTypeOpaqueWin32;
#elif defined(__linux__)
    static constexpr VkExternalMemoryHandleTypeFlagBits     vk_memory =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    static constexpr VkExternalSemaphoreHandleTypeFlagBits  vk_semaphore =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    static constexpr cudaExternalMemoryHandleType           cuda_memory =
        cudaExternalMemoryHandleTypeOpaqueFd;
    static constexpr cudaExternalSemaphoreHandleType        cuda_semaphore =
        cudaExternalSemaphoreHandleTypeOpaqueFd;
#else
    #error "VulkanOptixDenoiser: only Windows and Linux are supported (macOS uses MetalFX)."
#endif
};

// Native OS handle for an exported Vulkan memory or semaphore.
// On Win32 this is a HANDLE (void*); on Linux it's an int file
// descriptor. Stored as an opaque struct so callers don't have to
// #ifdef on every site that touches one.
struct NativeHandle {
#if defined(_WIN32)
    void* value = nullptr;            // HANDLE
    bool is_valid() const { return value != nullptr; }
#else
    int   value = -1;                 // file descriptor
    bool is_valid() const { return value >= 0; }
#endif
};

// Export helpers. Implemented in VulkanOptixDenoiser.cpp where the
// platform-specific Vulkan structs (VkMemoryGetWin32HandleInfoKHR vs
// VkMemoryGetFdInfoKHR) and entry-point function pointers can be
// resolved against the active VkDevice. Both helpers return VK_SUCCESS
// on success and leave `out` untouched on failure.
//
// Lifetime: the returned handle must be closed by the caller after
// CUDA imports it (CUDA dups the underlying object internally during
// cudaImportExternalMemory / cudaImportExternalSemaphore). On Windows
// use CloseHandle; on Linux use close(). Helpers in the .cpp wrap
// this for convenience.
VkResult ExportMemoryHandle(VkDevice         device,
                            VkDeviceMemory   memory,
                            NativeHandle*    out);

VkResult ExportSemaphoreHandle(VkDevice      device,
                               VkSemaphore   semaphore,
                               NativeHandle* out);

void CloseNativeHandle(NativeHandle& h);

}  // namespace pt::rhi::vk::external

#endif  // PT_ENABLE_OPTIX

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
#include <optix_stubs.h>

// Define the OptiX global function table exactly once per process.
// optix_stubs.h declares the table extern; this header defines it.
#include <optix_function_table_definition.h>

#include <atomic>
#include <cstring>

#if defined(_WIN32)
    #include <handleapi.h>      // CloseHandle
#else
    #include <unistd.h>         // close
#endif

namespace pt::rhi::vk {

// ---- Error logging helpers -----------------------------------------------
//
// All three runtimes (Vulkan / CUDA driver / CUDA runtime / OptiX) return
// distinct error code types. Wrap each in a small macro that logs + returns
// false on failure so the call sites stay readable. The macros short-circuit
// out of the enclosing function on failure.

#define VK_TRY(expr)                                                           \
    do {                                                                       \
        VkResult _r = (expr);                                                  \
        if (_r != VK_SUCCESS) {                                                \
            LOG_ERROR("VulkanOptixDenoiser: " #expr " failed (VkResult={})",   \
                      static_cast<int>(_r));                                   \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define CU_TRY(expr)                                                           \
    do {                                                                       \
        CUresult _r = (expr);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char* _msg = nullptr;                                        \
            cuGetErrorString(_r, &_msg);                                       \
            LOG_ERROR("VulkanOptixDenoiser: " #expr " failed: {}",             \
                      _msg ? _msg : "(no error string)");                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define CUDART_TRY(expr)                                                       \
    do {                                                                       \
        cudaError_t _r = (expr);                                               \
        if (_r != cudaSuccess) {                                               \
            LOG_ERROR("VulkanOptixDenoiser: " #expr " failed: {}",             \
                      cudaGetErrorString(_r));                                 \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define OPTIX_TRY(expr)                                                        \
    do {                                                                       \
        OptixResult _r = (expr);                                               \
        if (_r != OPTIX_SUCCESS) {                                             \
            LOG_ERROR("VulkanOptixDenoiser: " #expr " failed (OptixResult={})",\
                      static_cast<int>(_r));                                   \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace {

// optixInit must be called once per process. Guard with a flag so
// repeated denoiser construction (e.g. on cvar bounce) is safe.
std::atomic<bool> g_optix_initialized{false};

// Pick the format the path tracer's denoise_color matches (RGBA16F).
// All four images we shuttle (color_in, output, optional albedo,
// optional normal) are RGBA16F so they pack tightly into CUDA arrays
// and OptixImage2D::format = OPTIX_PIXEL_FORMAT_HALF4 lines up.
// OptiX denoiser input/output is RGBA16F. Vulkan buffers we feed it
// via cudaExternalMemoryGetMappedBuffer are tightly packed (no row
// padding) at 8 bytes per pixel.
constexpr OptixPixelFormat kOptixPixelFormat = OPTIX_PIXEL_FORMAT_HALF4;

// OptiX log callback. Hooks into the engine's logger so OptiX warnings
// surface in the regular log stream rather than stderr.
void OptixLogCb(unsigned int level, const char* tag, const char* msg, void*) {
    // OptiX uses 1=fatal, 2=error, 3=warning, 4=info; mirror to ours.
    switch (level) {
        case 1: LOG_ERROR("OptiX[{}]: {}", tag ? tag : "?", msg ? msg : ""); break;
        case 2: LOG_ERROR("OptiX[{}]: {}", tag ? tag : "?", msg ? msg : ""); break;
        case 3: LOG_WARN ("OptiX[{}]: {}", tag ? tag : "?", msg ? msg : ""); break;
        default:LOG_INFO ("OptiX[{}]: {}", tag ? tag : "?", msg ? msg : ""); break;
    }
}

}  // namespace

// ---- ExternalHandles helpers ---------------------------------------------
//
// Implemented here because they're only used by the OptiX denoiser path.
// Win32 uses NT handles via vkGetMemoryWin32HandleKHR; Linux uses FDs
// via vkGetMemoryFdKHR. Both functions are extension entry points,
// resolved per-VkDevice via vkGetDeviceProcAddr.

namespace external {

VkResult ExportMemoryHandle(VkDevice device, VkDeviceMemory memory,
                            NativeHandle* out) {
    if (out == nullptr) return VK_ERROR_VALIDATION_FAILED_EXT;
#if defined(_WIN32)
    auto pfn = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (pfn == nullptr) {
        LOG_ERROR("VulkanOptixDenoiser: vkGetMemoryWin32HandleKHR unresolved "
                  "(VK_KHR_external_memory_win32 not enabled?)");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkMemoryGetWin32HandleInfoKHR info{};
    info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    info.memory     = memory;
    info.handleType = HandleTypes::vk_memory;
    HANDLE h = nullptr;
    VkResult r = pfn(device, &info, &h);
    if (r == VK_SUCCESS) out->value = h;
    return r;
#else
    auto pfn = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    if (pfn == nullptr) {
        LOG_ERROR("VulkanOptixDenoiser: vkGetMemoryFdKHR unresolved "
                  "(VK_KHR_external_memory_fd not enabled?)");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkMemoryGetFdInfoKHR info{};
    info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    info.memory     = memory;
    info.handleType = HandleTypes::vk_memory;
    int fd = -1;
    VkResult r = pfn(device, &info, &fd);
    if (r == VK_SUCCESS) out->value = fd;
    return r;
#endif
}

VkResult ExportSemaphoreHandle(VkDevice device, VkSemaphore semaphore,
                               NativeHandle* out) {
    if (out == nullptr) return VK_ERROR_VALIDATION_FAILED_EXT;
#if defined(_WIN32)
    auto pfn = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (pfn == nullptr) {
        LOG_ERROR("VulkanOptixDenoiser: vkGetSemaphoreWin32HandleKHR unresolved");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkSemaphoreGetWin32HandleInfoKHR info{};
    info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    info.semaphore  = semaphore;
    info.handleType = HandleTypes::vk_semaphore;
    HANDLE h = nullptr;
    VkResult r = pfn(device, &info, &h);
    if (r == VK_SUCCESS) out->value = h;
    return r;
#else
    auto pfn = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
    if (pfn == nullptr) {
        LOG_ERROR("VulkanOptixDenoiser: vkGetSemaphoreFdKHR unresolved");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkSemaphoreGetFdInfoKHR info{};
    info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    info.semaphore  = semaphore;
    info.handleType = HandleTypes::vk_semaphore;
    int fd = -1;
    VkResult r = pfn(device, &info, &fd);
    if (r == VK_SUCCESS) out->value = fd;
    return r;
#endif
}

void CloseNativeHandle(NativeHandle& h) {
#if defined(_WIN32)
    if (h.value != nullptr) {
        CloseHandle(h.value);
        h.value = nullptr;
    }
#else
    if (h.value >= 0) {
        close(h.value);
        h.value = -1;
    }
#endif
}

}  // namespace external

// ---- VulkanOptixDenoiser ctor/dtor ---------------------------------------

VulkanOptixDenoiser::VulkanOptixDenoiser(VulkanDevice* device, Kind kind)
    : device_(device), kind_(kind) {
}

VulkanOptixDenoiser::~VulkanOptixDenoiser() {
    DestroyResources();
}

// ---- Init ------------------------------------------------------------------

bool VulkanOptixDenoiser::Init() {
    if (init_attempted_) return ready_;
    init_attempted_ = true;

    if (!InitOnce()) {
        LOG_ERROR("VulkanOptixDenoiser: InitOnce failed -- OptiX denoiser "
                  "unavailable for this session");
        DestroyResources();
        return false;
    }

    // Scratch + external image/semaphore allocation happens lazily on
    // the first Encode() call where we know the actual frame size.
    // Mark ready so VulkanDevice::Denoise stops re-Init-ing every frame.
    ready_ = true;
    LOG_INFO("VulkanOptixDenoiser: CUDA + OptiX context initialised "
             "(kind={}); scratch + external buffers sized lazily on "
             "first Encode",
             kind_ == Kind::HdrAov ? "HdrAov" : "Hdr");
    return true;
}

bool VulkanOptixDenoiser::InitOnce() {
    // ---- CUDA driver: init + context create ------------------------------
    CU_TRY(cuInit(0));

    int cuda_device_count = 0;
    CUDART_TRY(cudaGetDeviceCount(&cuda_device_count));
    if (cuda_device_count == 0) {
        LOG_ERROR("VulkanOptixDenoiser: no CUDA devices found");
        return false;
    }
    // For now assume device 0. A future improvement is to match the
    // VkPhysicalDevice deviceUUID against cudaDeviceProp::uuid so
    // multi-GPU iGPU+dGPU systems pick the same device for both APIs.
    CUdevice cu_dev = 0;
    CU_TRY(cuDeviceGet(&cu_dev, 0));

    char dev_name[256] = {};
    cuDeviceGetName(dev_name, sizeof(dev_name), cu_dev);
    LOG_INFO("VulkanOptixDenoiser: using CUDA device 0: {}", dev_name);

    // cuCtxCreate signature changed across toolkits:
    //   CUDA 12.x: macro -> cuCtxCreate_v3(CUcontext*, unsigned flags, CUdevice)
    //   CUDA 13.x: macro -> cuCtxCreate_v4(CUcontext*, CUctxCreateParams*, unsigned, CUdevice)
    // CMake's find_package(CUDAToolkit 12.0 ...) accepts both, so we
    // version-gate on CUDA_VERSION (>=13000 = 13.x) to compile against
    // either. nullptr for the v4 params struct -- we don't use exec
    // affinity / CIG / SM partitioning.
#if CUDA_VERSION >= 13000
    CU_TRY(cuCtxCreate(&cuda_ctx_, nullptr, 0, cu_dev));   // v4 (CUDA 13+)
#else
    CU_TRY(cuCtxCreate(&cuda_ctx_, 0, cu_dev));            // v3 (CUDA 12.x)
#endif
    CUDART_TRY(cudaStreamCreate(&cuda_stream_));

    // ---- OptiX: function table load + context + denoiser handle ----------
    if (!g_optix_initialized.load(std::memory_order_acquire)) {
        OPTIX_TRY(optixInit());
        g_optix_initialized.store(true, std::memory_order_release);
    }

    OptixDeviceContextOptions ctx_opts{};
    ctx_opts.logCallbackFunction = OptixLogCb;
    ctx_opts.logCallbackLevel    = 4;  // info+
    OPTIX_TRY(optixDeviceContextCreate(cuda_ctx_, &ctx_opts, &optix_ctx_));

    OptixDenoiserOptions opts{};
    OptixDenoiserModelKind model_kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    if (kind_ == Kind::HdrAov) {
        // AOV model: takes color + albedo + normal as guide layers.
        // The engine allocates the albedo + normal G-buffers and the
        // path tracer writes them every frame; here we just tell OptiX
        // we'll provide them so the denoiser builds the right state.
        opts.guideAlbedo = 1;
        opts.guideNormal = 1;
        model_kind       = OPTIX_DENOISER_MODEL_KIND_AOV;
        LOG_INFO("VulkanOptixDenoiser: kind=HdrAov -- using "
                 "OPTIX_DENOISER_MODEL_KIND_AOV with albedo + normal "
                 "guide layers (RGBA16F at vk::binding 17 / 16)");
    } else {
        // Plain HDR: no guide layers. OptiX uses only the noisy color
        // input + an internal HDR intensity scratch.
        opts.guideAlbedo = 0;
        opts.guideNormal = 0;
        LOG_INFO("VulkanOptixDenoiser: kind=Hdr -- using "
                 "OPTIX_DENOISER_MODEL_KIND_HDR (no AOV guides)");
    }
    OPTIX_TRY(optixDenoiserCreate(optix_ctx_,
                                  model_kind,
                                  &opts,
                                  &optix_denoiser_));
    return true;
}

// ---- ResizeScratch -------------------------------------------------------

bool VulkanOptixDenoiser::ResizeScratch(std::uint32_t w, std::uint32_t h) {
    OptixDenoiserSizes sizes{};
    OPTIX_TRY(optixDenoiserComputeMemoryResources(optix_denoiser_, w, h, &sizes));

    state_size_   = sizes.stateSizeInBytes;
    scratch_size_ = sizes.withoutOverlapScratchSizeInBytes;
    overlap_      = 0;  // no tiling; we always denoise the full frame

    // Free old scratch if any (resize path).
    if (state_buf_   != 0) { cudaFree(reinterpret_cast<void*>(state_buf_));   state_buf_   = 0; }
    if (scratch_buf_ != 0) { cudaFree(reinterpret_cast<void*>(scratch_buf_)); scratch_buf_ = 0; }
    if (intensity_buf_ != 0) { cudaFree(reinterpret_cast<void*>(intensity_buf_)); intensity_buf_ = 0; }

    void* p_state    = nullptr;
    void* p_scratch  = nullptr;
    void* p_intens   = nullptr;
    CUDART_TRY(cudaMalloc(&p_state,    state_size_));
    CUDART_TRY(cudaMalloc(&p_scratch,  scratch_size_));
    CUDART_TRY(cudaMalloc(&p_intens,   sizeof(float)));
    state_buf_     = reinterpret_cast<std::uint64_t>(p_state);
    scratch_buf_   = reinterpret_cast<std::uint64_t>(p_scratch);
    intensity_buf_ = reinterpret_cast<std::uint64_t>(p_intens);

    OPTIX_TRY(optixDenoiserSetup(optix_denoiser_, cuda_stream_,
                                 w, h,
                                 static_cast<CUdeviceptr>(state_buf_),   state_size_,
                                 static_cast<CUdeviceptr>(scratch_buf_), scratch_size_));
    LOG_INFO("VulkanOptixDenoiser: scratch sized for {}x{} (state={}MB, scratch={}MB)",
             w, h, state_size_   / (1024 * 1024),
                   scratch_size_ / (1024 * 1024));
    return true;
}

// ---- ResizeExternalBuffers -----------------------------------------------

namespace {

// Find a Vulkan memory type matching the type bits + property mask.
std::uint32_t FindMemoryType(VkPhysicalDevice phys, std::uint32_t type_bits,
                             VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

}  // namespace

// VkBuffers (not VkImages) backing the OptiX denoiser inputs/outputs.
// OptixImage2D::data takes a CUdeviceptr -- cudaExternalMemoryGetMappedBuffer
// (on a buffer-mode external memory import) returns exactly that, while
// the image-mode import returns cudaArrays which OptiX can't accept.
bool VulkanOptixDenoiser::ResizeExternalBuffers(std::uint32_t w, std::uint32_t h) {
    DestroyExternalBuffer(0);
    DestroyExternalBuffer(1);
    DestroyExternalBuffer(2);  // albedo (no-op when not previously allocated)
    DestroyExternalBuffer(3);  // normal (no-op when not previously allocated)

    VkDevice dev = device_->RawDevice();
    VkPhysicalDevice phys = device_->RawPhysicalDevice();

    const VkDeviceSize buf_size =
        static_cast<VkDeviceSize>(w) * h * kBytesPerPixel;

    auto build_one = [&](ExternalBuffer& out, const char* label) -> bool {
        VkExternalMemoryBufferCreateInfo embci{};
        embci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        embci.handleTypes = external::HandleTypes::vk_memory;

        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext       = &embci;
        bci.size        = buf_size;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(dev, &bci, nullptr, &out.buffer));
        out.size = buf_size;

        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, out.buffer, &mr);

        VkExportMemoryAllocateInfo emai{};
        emai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        emai.handleTypes = external::HandleTypes::vk_memory;

        VkMemoryDedicatedAllocateInfo mdai{};
        mdai.sType   = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        mdai.buffer  = out.buffer;
        mdai.pNext   = &emai;

        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext           = &mdai;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = FindMemoryType(phys, mr.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) {
            LOG_ERROR("VulkanOptixDenoiser({}): no DEVICE_LOCAL memory type "
                      "matches typeBits=0x{:x}", label, mr.memoryTypeBits);
            return false;
        }
        VK_TRY(vkAllocateMemory(dev, &mai, nullptr, &out.memory));
        VK_TRY(vkBindBufferMemory(dev, out.buffer, out.memory, 0));

        external::NativeHandle nh;
        if (external::ExportMemoryHandle(dev, out.memory, &nh) != VK_SUCCESS ||
            !nh.is_valid()) {
            LOG_ERROR("VulkanOptixDenoiser({}): ExportMemoryHandle failed", label);
            return false;
        }

        cudaExternalMemoryHandleDesc emdesc{};
        std::memset(&emdesc, 0, sizeof(emdesc));
        emdesc.type = external::HandleTypes::cuda_memory;
#if defined(_WIN32)
        emdesc.handle.win32.handle = nh.value;
#else
        emdesc.handle.fd           = nh.value;
#endif
        emdesc.size  = mr.size;
        emdesc.flags = cudaExternalMemoryDedicated;
        cudaError_t cerr = cudaImportExternalMemory(&out.cuda_ext, &emdesc);
        external::CloseNativeHandle(nh);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaImportExternalMemory: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }

        // Map as a buffer -> CUdeviceptr. This is what OptixImage2D
        // wants. The pointer's lifetime is tied to cuda_ext; freeing
        // cuda_ext (cudaDestroyExternalMemory) frees the mapping too.
        cudaExternalMemoryBufferDesc bdesc{};
        std::memset(&bdesc, 0, sizeof(bdesc));
        bdesc.offset = 0;
        bdesc.size   = mr.size;
        bdesc.flags  = 0;
        void* dev_ptr = nullptr;
        cerr = cudaExternalMemoryGetMappedBuffer(&dev_ptr, out.cuda_ext, &bdesc);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaExternalMemoryGetMappedBuffer: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }
        out.cuda_ptr = reinterpret_cast<std::uint64_t>(dev_ptr);
        return true;
    };

    if (!build_one(buf_color_in_, "color_in")) return false;
    if (!build_one(buf_output_,   "output"))   return false;

    int aov_count = 0;
    if (kind_ == Kind::HdrAov) {
        // AOV guide layers: same external-memory machinery as color_in
        // and output (linear DEVICE_LOCAL VkBuffer + cudaImportExternal-
        // Memory + GetMappedBuffer). OptiX reads these as half4 RGBA in
        // OptixDenoiserGuideLayer::albedo / .normal during Invoke.
        if (!build_one(buf_albedo_, "albedo")) return false;
        if (!build_one(buf_normal_, "normal")) return false;
        aov_count = 2;
    }

    LOG_INFO("VulkanOptixDenoiser: external buffers sized for {}x{} "
             "({} x {}KB, dedicated allocation, imported to CUDA as "
             "linear CUdeviceptrs)", w, h, 2 + aov_count, buf_size / 1024);
    return true;
}

void VulkanOptixDenoiser::DestroyExternalBuffer(int slot) {
    // Slot semantics:
    //   0 = buf_color_in_  (always allocated)
    //   1 = buf_output_    (always allocated)
    //   2 = buf_albedo_    (HdrAov only; null on Hdr -- safe to call)
    //   3 = buf_normal_    (HdrAov only; null on Hdr -- safe to call)
    ExternalBuffer* pe = nullptr;
    switch (slot) {
        case 0: pe = &buf_color_in_; break;
        case 1: pe = &buf_output_;   break;
        case 2: pe = &buf_albedo_;   break;
        case 3: pe = &buf_normal_;   break;
        default:
            LOG_WARN("VulkanOptixDenoiser::DestroyExternalBuffer: "
                     "unexpected slot={}", slot);
            return;
    }
    ExternalBuffer& e = *pe;
    // CUDA mapping freed implicitly when cudaDestroyExternalMemory runs.
    if (e.cuda_ext != nullptr) { cudaDestroyExternalMemory(e.cuda_ext); e.cuda_ext = nullptr; }
    e.cuda_ptr = 0;

    VkDevice dev = device_ ? device_->RawDevice() : VK_NULL_HANDLE;
    if (dev != VK_NULL_HANDLE) {
        if (e.buffer != VK_NULL_HANDLE) { vkDestroyBuffer(dev, e.buffer, nullptr); e.buffer = VK_NULL_HANDLE; }
        if (e.memory != VK_NULL_HANDLE) { vkFreeMemory(dev, e.memory, nullptr); e.memory = VK_NULL_HANDLE; }
    }
    e.size = 0;
}

// ---- ResizeExternalSemaphores --------------------------------------------

bool VulkanOptixDenoiser::ResizeExternalSemaphores() {
    DestroyExternalSemaphores();

    VkDevice dev = device_->RawDevice();

    auto make_one = [&](VkSemaphore& vk_sem, cudaExternalSemaphore_t& cu_sem) -> bool {
        VkExportSemaphoreCreateInfo esci{};
        esci.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        esci.handleTypes = external::HandleTypes::vk_semaphore;

        VkSemaphoreTypeCreateInfo stci{};
        stci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue  = 0;
        stci.pNext         = &esci;

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sci.pNext = &stci;
        VK_TRY(vkCreateSemaphore(dev, &sci, nullptr, &vk_sem));

        external::NativeHandle nh;
        if (external::ExportSemaphoreHandle(dev, vk_sem, &nh) != VK_SUCCESS ||
            !nh.is_valid()) {
            LOG_ERROR("VulkanOptixDenoiser: ExportSemaphoreHandle failed");
            return false;
        }

        cudaExternalSemaphoreHandleDesc desc{};
        std::memset(&desc, 0, sizeof(desc));
        desc.type = external::HandleTypes::cuda_semaphore;
#if defined(_WIN32)
        desc.handle.win32.handle = nh.value;
#else
        desc.handle.fd           = nh.value;
#endif
        // For timeline semaphores no extra flag is needed -- the handle
        // type itself (TimelineSemaphoreWin32 / TimelineSemaphoreFd) is
        // what tells CUDA the imported VkSemaphore is timeline-typed.
        // ExternalHandles.h's HandleTypes::cuda_semaphore picks the
        // right one per-platform.
        desc.flags = 0;
        cudaError_t cerr = cudaImportExternalSemaphore(&cu_sem, &desc);
        external::CloseNativeHandle(nh);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser: cudaImportExternalSemaphore: {}",
                      cudaGetErrorString(cerr));
            return false;
        }
        return true;
    };

    if (!make_one(sem_vk_to_cuda_, cuda_sem_vk_to_cuda_)) return false;
    if (!make_one(sem_cuda_to_vk_, cuda_sem_cuda_to_vk_)) return false;
    timeline_counter_ = 0;
    return true;
}

void VulkanOptixDenoiser::DestroyExternalSemaphores() {
    if (cuda_sem_vk_to_cuda_ != nullptr) { cudaDestroyExternalSemaphore(cuda_sem_vk_to_cuda_); cuda_sem_vk_to_cuda_ = nullptr; }
    if (cuda_sem_cuda_to_vk_ != nullptr) { cudaDestroyExternalSemaphore(cuda_sem_cuda_to_vk_); cuda_sem_cuda_to_vk_ = nullptr; }
    VkDevice dev = device_ ? device_->RawDevice() : VK_NULL_HANDLE;
    if (dev != VK_NULL_HANDLE) {
        if (sem_vk_to_cuda_ != VK_NULL_HANDLE) { vkDestroySemaphore(dev, sem_vk_to_cuda_, nullptr); sem_vk_to_cuda_ = VK_NULL_HANDLE; }
        if (sem_cuda_to_vk_ != VK_NULL_HANDLE) { vkDestroySemaphore(dev, sem_cuda_to_vk_, nullptr); sem_cuda_to_vk_ = VK_NULL_HANDLE; }
    }
    timeline_counter_ = 0;
}

// ---- EnsureCommandPool ---------------------------------------------------

bool VulkanOptixDenoiser::EnsureCommandPool() {
    if (private_pool_ != VK_NULL_HANDLE) return true;
    VkDevice dev = device_->RawDevice();
    VkCommandPoolCreateInfo pci{};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = device_->GraphicsQueueFamily();
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_TRY(vkCreateCommandPool(dev, &pci, nullptr, &private_pool_));
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = private_pool_;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kFramesInFlight;
    VK_TRY(vkAllocateCommandBuffers(dev, &cbai, private_cb_out_));
    return true;
}

void VulkanOptixDenoiser::DestroyCommandPool() {
    VkDevice dev = device_ ? device_->RawDevice() : VK_NULL_HANDLE;
    if (dev != VK_NULL_HANDLE && private_pool_ != VK_NULL_HANDLE) {
        // VkCommandBuffers are freed implicitly with the pool.
        vkDestroyCommandPool(dev, private_pool_, nullptr);
    }
    private_pool_ = VK_NULL_HANDLE;
    for (auto& cb : private_cb_out_) cb = VK_NULL_HANDLE;
    private_cb_index_ = 0;
}

// ---- Encode --------------------------------------------------------------
//
// Per-frame flow when r_denoiser optix_hdr is active. v1 sync model
// favours simplicity over peak perf: vkQueueWaitIdle at the start of
// SubmitPostMain + cudaStreamSynchronize before submitting the
// private cb. CPU stalls per frame for the OptiX latency (~2-4 ms
// at 720p HDR) but no risk of cross-API deadlock from optix calls
// with hidden CPU-side synchronisation. Async via timeline
// semaphores is wired up at the API layer (sem_vk_to_cuda_,
// sem_cuda_to_vk_, RequestExtraSubmitSignal) but not used in the
// per-frame path -- ready to re-enable as a future commit.
//
//   Frame N tick:
//     1. Engine path-tracer dispatch records into engine cb (already
//        done before VulkanDevice::Denoise dispatched here).
//     2. Encode appends to the engine cb:
//          - barrier d.color_in: GENERAL -> TRANSFER_SRC
//          - vkCmdCopyImageToBuffer(d.color_in -> buf_color_in_)
//          - barrier d.color_in: TRANSFER_SRC -> GENERAL (path tracer
//            wants GENERAL again next frame)
//     3. Encode records (without submitting) the private cb:
//          - barrier d.output: GENERAL -> TRANSFER_DST
//          - vkCmdCopyBufferToImage(buf_output_ -> d.output)
//          - barrier d.output: TRANSFER_DST -> GENERAL (compute read)
//          - barrier swapchain: UNDEFINED -> GENERAL (compute write)
//          - VulkanDevice::EncodeDenoiseFinalize (ACES + sRGB tonemap)
//          - barrier swapchain: GENERAL -> PRESENT_SRC
//        Saves the cb slot + frame index for SubmitPostMain.
//     4. Encode returns. Engine continues recording into engine cb,
//        then VulkanDevice::Submit submits engine cb.
//     5. SubmitPostMain (called by Submit AFTER engine cb submit):
//          - vkQueueWaitIdle (engine cb done -> buf_color_in_ has data)
//          - optixDenoiserComputeIntensity (HDR model needs intensity)
//          - optixDenoiserInvoke (buf_color_in_.cuda_ptr ->
//                                 buf_output_.cuda_ptr)
//          - cudaStreamSynchronize (block until denoise done)
//          - vkQueueSubmit private cb (no semaphore waits -- queue
//            order + the CPU stall above already serialise everything)
//     6. Engine's vkQueuePresentKHR runs after SubmitPostMain returns,
//        sees the tonemapped swapchain.
//
// The future async-with-semaphores variant of step 4-5 is documented
// in 7fdbdc0's commit message: it wants RequestExtraSubmitSignal
// to chain a timeline signal onto the engine submit, then CUDA's
// async wait on that signal lets ComputeIntensity / Invoke run
// async-on-stream, then the private cb waits on the cuda->vk signal.
// Deferred until we've mapped which OptiX calls have hidden CPU sync
// (optixDenoiserComputeIntensity is one; full audit pending).

void VulkanOptixDenoiser::Encode(VkCommandBuffer cb,
                                 const Device::DenoiseDesc& d) {
    if (!ready_) return;
    if (cb == VK_NULL_HANDLE)   return;
    if (d.color_in.id == 0 || d.output.id == 0) return;

    VkDevice dev = device_->RawDevice();

    // ---- Lazy first-time setup: scratch + buffers + sems + private pool --
    //
    // Size pulled from the engine's d.color_in extent so we automatically
    // match whatever the engine allocated for the path tracer (the engine
    // sizes the G-buffer to the swapchain). Window resize will currently
    // hit a stale-size fast path -- proper resize handling is a small
    // follow-up (free + rebuild scratch / buffers / sems on extent change,
    // matching the existing OptiX state-buffer-keyed-on-WxH pattern).
    VkExtent2D src_extent = device_->LookupImageExtent(d.color_in);
    if (src_extent.width == 0 || src_extent.height == 0) {
        LOG_WARN("VulkanOptixDenoiser::Encode: color_in extent lookup returned 0x0");
        return;
    }
    const std::uint32_t target_w = src_extent.width;
    const std::uint32_t target_h = src_extent.height;

    if (target_w != cached_w_ || target_h != cached_h_) {
        if (!ResizeScratch(target_w, target_h)         ||
            !ResizeExternalBuffers(target_w, target_h) ||
            !ResizeExternalSemaphores()                ||
            !EnsureCommandPool()) {
            LOG_ERROR("VulkanOptixDenoiser: lazy first-time setup failed; "
                      "marking denoiser un-ready");
            ready_ = false;
            return;
        }
        cached_w_ = target_w;
        cached_h_ = target_h;
        LOG_INFO("VulkanOptixDenoiser: per-frame denoise dispatch active "
                 "for {}x{}", target_w, target_h);
    }

    // ---- Resolve engine VkImages ---------------------------------------
    VkImage src_img      = device_->LookupImage(d.color_in);
    VkImage dst_img      = device_->LookupImage(d.output);
    VkImage swap_img     = device_->LookupImage(d.final_output);
    VkImage albedo_img   = VK_NULL_HANDLE;
    VkImage normal_img   = VK_NULL_HANDLE;
    if (kind_ == Kind::HdrAov) {
        albedo_img = device_->LookupImage(d.albedo_in);
        normal_img = device_->LookupImage(d.normal_in);
        if (albedo_img == VK_NULL_HANDLE || normal_img == VK_NULL_HANDLE) {
            LOG_WARN("VulkanOptixDenoiser::Encode(HdrAov): AOV image lookup miss "
                     "(albedo={} normal={}); skipping frame",
                     static_cast<void*>(albedo_img),
                     static_cast<void*>(normal_img));
            return;
        }
    }
    if (src_img == VK_NULL_HANDLE || dst_img == VK_NULL_HANDLE) {
        LOG_WARN("VulkanOptixDenoiser::Encode: image lookup miss "
                 "(src={} dst={})", static_cast<void*>(src_img),
                                    static_cast<void*>(dst_img));
        return;
    }
    // swap_img may be VK_NULL_HANDLE if engine didn't pass final_output
    // (older callers). In that case we skip the blit-to-swapchain step
    // and the image stays noisy on screen, but the OptiX denoise at
    // least lands cleanly in d.output for any downstream finalize.

    timeline_counter_++;
    const std::uint64_t this_frame = timeline_counter_;

    // ---- Step 2: append to engine cb -- copy d.color_in -> buf_color_in_ -
    {
        VkImageMemoryBarrier toSrc{};
        toSrc.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        toSrc.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image         = src_img;
        toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.layerCount = 1;
        toSrc.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;  // tightly packed
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent       = { cached_w_, cached_h_, 1 };
        vkCmdCopyImageToBuffer(cb, src_img,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buf_color_in_.buffer,
                               1, &region);

        // Restore d.color_in to GENERAL so the engine's path-tracer
        // dispatch on the next frame sees the layout it expects.
        VkImageMemoryBarrier backToGeneral{};
        backToGeneral.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        backToGeneral.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        backToGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        backToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        backToGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        backToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneral.image         = src_img;
        backToGeneral.subresourceRange = toSrc.subresourceRange;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &backToGeneral);
    }

    // ---- Step 2b: AOV guide-layer copies (HdrAov only) ------------------
    // For OPTIX_DENOISER_MODEL_KIND_AOV the guide layer wants linear
    // CUdeviceptrs for albedo + normal. Same machinery as color_in:
    // GENERAL -> TRANSFER_SRC, vkCmdCopyImageToBuffer, back to GENERAL.
    // Two images means two barrier-pairs; we batch each image's
    // barriers into a single vkCmdPipelineBarrier rather than a single
    // multi-barrier call across both images, because the source layouts
    // are independent per-image and barrier batching across images
    // doesn't actually save anything on NVIDIA's driver beyond what
    // the engine's per-image batching already achieves.
    if (kind_ == Kind::HdrAov) {
        auto copy_aov_image_to_buffer = [&](VkImage img, VkBuffer buf, const char* label) {
            VkImageMemoryBarrier toSrc{};
            toSrc.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            toSrc.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image         = img;
            toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toSrc.subresourceRange.layerCount = 1;
            toSrc.subresourceRange.levelCount = 1;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &toSrc);

            VkBufferImageCopy region{};
            region.bufferOffset      = 0;
            region.bufferRowLength   = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { cached_w_, cached_h_, 1 };
            vkCmdCopyImageToBuffer(cb, img,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   buf, 1, &region);

            VkImageMemoryBarrier back{};
            back.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            back.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            back.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            back.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            back.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            back.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            back.image         = img;
            back.subresourceRange = toSrc.subresourceRange;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &back);
            (void)label;
        };
        copy_aov_image_to_buffer(albedo_img, buf_albedo_.buffer, "albedo");
        copy_aov_image_to_buffer(normal_img, buf_normal_.buffer, "normal");
    }

    // CUDA + OptiX work is deferred to SubmitPostMain (called from
    // VulkanDevice::Submit AFTER the engine cb is submitted). Doing it
    // here would deadlock: optixDenoiserComputeIntensity has implicit
    // synchronisation on first call (probably JIT-related), and at
    // Encode time the engine cb is still being recorded -- we can't
    // wait for its completion from inside the recording itself.

    // ---- Step 5: record private cb that copies buf_output_ -> d.output ---
    // (Submitted later by SubmitPostMain(), called from VulkanDevice::Submit
    // AFTER the main engine cb is submitted -- queue-order serialization
    // would otherwise deadlock on the engine cb signalling the timeline
    // our private cb's wait depends on.)
    const int  pcb_slot = private_cb_index_;
    VkCommandBuffer pcb = private_cb_out_[pcb_slot];
    private_cb_index_   = (private_cb_index_ + 1) % kFramesInFlight;
    vkResetCommandBuffer(pcb, 0);
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(pcb, &bi) != VK_SUCCESS) {
            LOG_WARN("VulkanOptixDenoiser::Encode: vkBeginCommandBuffer(private) failed");
            return;
        }

        VkImageMemoryBarrier toDst{};
        toDst.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // GENERAL is the actual layout on entry: VulkanDevice::CreateTexture
        // submits a one-shot UNDEFINED -> GENERAL barrier at allocation
        // time, and the prior frame's DenoiseFinalize compute pass left
        // it in GENERAL (it was the SHADER_READ source of that pass).
        // Using UNDEFINED here would (a) trip validation on a "wrong
        // current layout" check, and (b) skip synchronisation with the
        // prior frame's compute read -- the queue's vkQueueWaitIdle in
        // SubmitPostMain already covers (b) but the sync intent is what
        // the barrier should express.
        toDst.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        toDst.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;   // prior finalize pass
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image         = dst_img;
        toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toDst.subresourceRange.layerCount = 1;
        toDst.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(pcb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // prior finalize pass read d.output
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toDst);

        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent       = { cached_w_, cached_h_, 1 };
        vkCmdCopyBufferToImage(pcb, buf_output_.buffer, dst_img,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &region);

        // d.output: TRANSFER_DST -> GENERAL (so DenoiseFinalize can
        // shader-read it as a storage image).
        VkImageMemoryBarrier outToGeneral{};
        outToGeneral.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        outToGeneral.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        outToGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        outToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        outToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        outToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outToGeneral.image         = dst_img;
        outToGeneral.subresourceRange = toDst.subresourceRange;
        vkCmdPipelineBarrier(pcb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &outToGeneral);

        // ---- DenoiseFinalize: ACES + sRGB tonemap to swapchain ----
        // Reuses the SVGF DenoiseFinalize.slang compute kernel via the
        // VulkanDevice::EncodeDenoiseFinalize pass-through. Lazy-inits
        // VulkanNrdDenoiser if SVGF isn't yet active; only the finalize
        // pipeline is built (the SVGF history textures stay unallocated
        // unless the user actually picks svgf_*).
        if (swap_img != VK_NULL_HANDLE) {
            // Swapchain: UNDEFINED -> GENERAL for compute write.
            // Engine cb left it in PRESENT_SRC; UNDEFINED is spec-correct
            // because we overwrite every pixel.
            VkImageMemoryBarrier swapToGeneral{};
            swapToGeneral.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            swapToGeneral.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            swapToGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            swapToGeneral.srcAccessMask = 0;
            swapToGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            swapToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapToGeneral.image         = swap_img;
            swapToGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            swapToGeneral.subresourceRange.layerCount = 1;
            swapToGeneral.subresourceRange.levelCount = 1;
            vkCmdPipelineBarrier(pcb,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &swapToGeneral);

            VkImageView dst_view  = device_->LookupImageView(d.output);
            VkImageView swap_view = device_->LookupImageView(d.final_output);
            VkBuffer    exp_buf   = device_->LookupBuffer(d.exposure_state);
            if (dst_view != VK_NULL_HANDLE && swap_view != VK_NULL_HANDLE &&
                exp_buf  != VK_NULL_HANDLE) {
                device_->EncodeDenoiseFinalize(pcb, dst_view, swap_view, exp_buf,
                                               cached_w_, cached_h_, d.hdr_pipeline);
            } else {
                LOG_WARN("VulkanOptixDenoiser::Encode: finalize lookup miss "
                         "(dst_view={} swap_view={} exp_buf={}); skipping tonemap",
                         static_cast<void*>(dst_view),
                         static_cast<void*>(swap_view),
                         static_cast<void*>(exp_buf));
            }

            // Swapchain back to PRESENT_SRC for vkQueuePresentKHR.
            VkImageMemoryBarrier swapToPres{};
            swapToPres.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            swapToPres.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            swapToPres.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            swapToPres.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            swapToPres.dstAccessMask = 0;
            swapToPres.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapToPres.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapToPres.image         = swap_img;
            swapToPres.subresourceRange = swapToGeneral.subresourceRange;
            vkCmdPipelineBarrier(pcb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr, 1, &swapToPres);
        }

        vkEndCommandBuffer(pcb);
    }

    // Submit deferred to SubmitPostMain(); save state for that call.
    pending_post_main_  = true;
    pending_pcb_slot_   = pcb_slot;
    pending_wait_value_ = this_frame;
    (void)dev;
}

void VulkanOptixDenoiser::DrainCuda() {
    if (cuda_stream_ != nullptr) {
        cudaStreamSynchronize(cuda_stream_);
    }
}

void VulkanOptixDenoiser::SubmitPostMain() {
    if (!pending_post_main_) return;
    pending_post_main_ = false;

    // V1 sync model: wait for engine cb (and prior private cbs) to
    // finish so buf_color_in_ has the path-tracer's output, run the
    // OptiX denoise on the CUDA stream, CPU-sync to wait for it, then
    // submit the private output-copy cb with no semaphore waits. CPU
    // stalls per frame for the OptiX latency (~2-4 ms at 1080p HDR
    // model), but no risk of cross-API deadlock. Async via timeline
    // semaphores is a future optimisation -- the timeline-semaphore
    // resources are still allocated by ResizeExternalSemaphores but
    // not used in the per-frame sync path here.
    VkResult vr = vkQueueWaitIdle(device_->RawGraphicsQueue());
    if (vr != VK_SUCCESS) {
        LOG_WARN("VulkanOptixDenoiser::SubmitPostMain: vkQueueWaitIdle "
                 "(VkResult={}); skipping OptiX work this frame",
                 static_cast<int>(vr));
        return;
    }

    // ---- CUDA: schedule + sync the OptiX denoise ----
    OptixImage2D color_in{};
    color_in.data             = static_cast<CUdeviceptr>(buf_color_in_.cuda_ptr);
    color_in.width            = cached_w_;
    color_in.height           = cached_h_;
    color_in.rowStrideInBytes = cached_w_ * kBytesPerPixel;
    color_in.pixelStrideInBytes = kBytesPerPixel;
    color_in.format           = kOptixPixelFormat;

    OptixImage2D output{};
    output.data             = static_cast<CUdeviceptr>(buf_output_.cuda_ptr);
    output.width            = cached_w_;
    output.height           = cached_h_;
    output.rowStrideInBytes = cached_w_ * kBytesPerPixel;
    output.pixelStrideInBytes = kBytesPerPixel;
    output.format           = kOptixPixelFormat;

    OptixResult oresult = optixDenoiserComputeIntensity(
        optix_denoiser_, cuda_stream_,
        &color_in,
        static_cast<CUdeviceptr>(intensity_buf_),
        static_cast<CUdeviceptr>(scratch_buf_),
        scratch_size_);
    if (oresult != OPTIX_SUCCESS) {
        LOG_ERROR("VulkanOptixDenoiser::SubmitPostMain: optixDenoiserComputeIntensity "
                  "(OptixResult={})", static_cast<int>(oresult));
        return;
    }

    OptixDenoiserParams params{};
    std::memset(&params, 0, sizeof(params));
    params.hdrIntensity = static_cast<CUdeviceptr>(intensity_buf_);
    params.blendFactor  = 0.0f;

    OptixDenoiserGuideLayer guide_layer{};
    std::memset(&guide_layer, 0, sizeof(guide_layer));
    if (kind_ == Kind::HdrAov) {
        // Albedo guide: linear-RGB at primary hit. OptiX uses it to
        // preserve diffuse-color edges that the noisy color buffer
        // alone can't disambiguate (e.g. wallpaper-on-shadow boundary
        // looks like noise to the color denoiser).
        guide_layer.albedo.data             = static_cast<CUdeviceptr>(buf_albedo_.cuda_ptr);
        guide_layer.albedo.width            = cached_w_;
        guide_layer.albedo.height           = cached_h_;
        guide_layer.albedo.rowStrideInBytes = cached_w_ * kBytesPerPixel;
        guide_layer.albedo.pixelStrideInBytes = kBytesPerPixel;
        guide_layer.albedo.format           = kOptixPixelFormat;
        // Normal guide: world-space surface normal at primary hit.
        // Used to preserve geometric edges + suppress filtering across
        // surface-orientation discontinuities (silhouettes / corners).
        guide_layer.normal.data             = static_cast<CUdeviceptr>(buf_normal_.cuda_ptr);
        guide_layer.normal.width            = cached_w_;
        guide_layer.normal.height           = cached_h_;
        guide_layer.normal.rowStrideInBytes = cached_w_ * kBytesPerPixel;
        guide_layer.normal.pixelStrideInBytes = kBytesPerPixel;
        guide_layer.normal.format           = kOptixPixelFormat;
    }

    OptixDenoiserLayer layer{};
    std::memset(&layer, 0, sizeof(layer));
    layer.input  = color_in;
    layer.output = output;

    oresult = optixDenoiserInvoke(
        optix_denoiser_, cuda_stream_,
        &params,
        static_cast<CUdeviceptr>(state_buf_), state_size_,
        &guide_layer,
        &layer, 1,
        /*inputOffsetX*/ 0, /*inputOffsetY*/ 0,
        static_cast<CUdeviceptr>(scratch_buf_), scratch_size_);
    if (oresult != OPTIX_SUCCESS) {
        LOG_ERROR("VulkanOptixDenoiser::SubmitPostMain: optixDenoiserInvoke "
                  "(OptixResult={})", static_cast<int>(oresult));
        return;
    }

    cudaError_t cerr = cudaStreamSynchronize(cuda_stream_);
    if (cerr != cudaSuccess) {
        LOG_ERROR("VulkanOptixDenoiser::SubmitPostMain: cudaStreamSynchronize: {}",
                  cudaGetErrorString(cerr));
        return;
    }

    // ---- Submit private output-copy cb with no waits ----
    VkCommandBuffer pcb = private_cb_out_[pending_pcb_slot_];
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &pcb;
    si.signalSemaphoreCount = 0;
    si.waitSemaphoreCount   = 0;
    if (vkQueueSubmit(device_->RawGraphicsQueue(), 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        LOG_WARN("VulkanOptixDenoiser::SubmitPostMain: vkQueueSubmit(private) failed");
    }
}

// ---- DestroyResources -----------------------------------------------------

void VulkanOptixDenoiser::DestroyResources() {
    // Ensure all in-flight GPU work has drained before tearing down
    // any external memory; otherwise the driver may still be reading
    // an imported handle when we destroy it.
    if (cuda_stream_ != nullptr) {
        cudaStreamSynchronize(cuda_stream_);
    }

    DestroyExternalSemaphores();
    DestroyCommandPool();
    DestroyExternalBuffer(0);
    DestroyExternalBuffer(1);
    DestroyExternalBuffer(2);  // albedo (no-op on Hdr, valid on HdrAov)
    DestroyExternalBuffer(3);  // normal (no-op on Hdr, valid on HdrAov)

    if (state_buf_     != 0) { cudaFree(reinterpret_cast<void*>(state_buf_));     state_buf_     = 0; }
    if (scratch_buf_   != 0) { cudaFree(reinterpret_cast<void*>(scratch_buf_));   scratch_buf_   = 0; }
    if (intensity_buf_ != 0) { cudaFree(reinterpret_cast<void*>(intensity_buf_)); intensity_buf_ = 0; }
    state_size_   = 0;
    scratch_size_ = 0;

    if (optix_denoiser_ != nullptr) {
        optixDenoiserDestroy(optix_denoiser_);
        optix_denoiser_ = nullptr;
    }
    if (optix_ctx_ != nullptr) {
        optixDeviceContextDestroy(optix_ctx_);
        optix_ctx_ = nullptr;
    }
    if (cuda_stream_ != nullptr) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    if (cuda_ctx_ != nullptr) {
        cuCtxDestroy(cuda_ctx_);
        cuda_ctx_ = nullptr;
    }

    cached_w_ = cached_h_ = 0;
    ready_    = false;
}

}  // namespace pt::rhi::vk

#endif  // PT_ENABLE_OPTIX

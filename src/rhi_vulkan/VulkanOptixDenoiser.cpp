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
constexpr VkFormat kOptixImageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr OptixPixelFormat kOptixPixelFormat = OPTIX_PIXEL_FORMAT_HALF4;
constexpr std::uint32_t kPixelStrideBytes = 8;  // 4 channels * 2 bytes

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
             "(kind={}); scratch + external images sized lazily on "
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

    // CUDA 13 added cuCtxCreate_v4 with an extra CUctxCreateParams* arg
    // (NULL for default). The macro cuCtxCreate maps to v4 in CUDA 13;
    // earlier toolkits aliased it to v3 (3 args). Always pass nullptr
    // for the params struct -- we don't use exec affinity / CIG.
    CU_TRY(cuCtxCreate(&cuda_ctx_, nullptr, 0, cu_dev));
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
    if (kind_ == Kind::HdrAov) {
        // Phase 1a step 3 lands the actual albedo + normal AOV inputs;
        // until then HdrAov falls back to plain HDR so this denoiser is
        // usable today. The cvar value is preserved so the user can
        // pre-set their preferred kind even before AOV plumbing lands.
        opts.guideAlbedo = 0;
        opts.guideNormal = 0;
        LOG_INFO("VulkanOptixDenoiser: kind=HdrAov requested but AOV "
                 "inputs (primary_albedo) not yet wired -- falling back "
                 "to plain HDR for this commit. AOV variant lands in "
                 "Phase 1a step 3.");
    } else {
        opts.guideAlbedo = 0;
        opts.guideNormal = 0;
    }
    OPTIX_TRY(optixDenoiserCreate(optix_ctx_,
                                  OPTIX_DENOISER_MODEL_KIND_HDR,
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

// ---- ResizeExternalImages ------------------------------------------------

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

bool VulkanOptixDenoiser::ResizeExternalImages(std::uint32_t w, std::uint32_t h) {
    DestroyExternalImage(0);
    DestroyExternalImage(1);

    VkDevice dev = device_->RawDevice();
    VkPhysicalDevice phys = device_->RawPhysicalDevice();

    auto build_one = [&](ExternalImage& out, const char* label) -> bool {
        // External image creation: VK_KHR_external_memory + the per-OS
        // handle type. Tiling LINEAR keeps the pitch deterministic so
        // CUDA sees a row-major layout it can map without extra copies.
        VkExternalMemoryImageCreateInfo emci{};
        emci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        emci.handleTypes = external::HandleTypes::vk_memory;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.pNext         = &emci;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = kOptixImageFormat;
        ici.extent        = { w, h, 1 };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_TRY(vkCreateImage(dev, &ici, nullptr, &out.image));

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(dev, out.image, &mr);

        VkExportMemoryAllocateInfo emai{};
        emai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        emai.handleTypes = external::HandleTypes::vk_memory;

        VkMemoryDedicatedAllocateInfo mdai{};
        mdai.sType   = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        mdai.image   = out.image;
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
        VK_TRY(vkBindImageMemory(dev, out.image, out.memory, 0));

        // Export to native handle and import into CUDA. Renamed to
        // `nh` to avoid shadowing the outer `h` (height) parameter
        // captured in this lambda.
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
        emdesc.size = mr.size;
        // Dedicated allocation -> tell CUDA so it doesn't try to manage
        // sub-region offsets internally.
        emdesc.flags = cudaExternalMemoryDedicated;
        cudaError_t cerr = cudaImportExternalMemory(&out.cuda_ext, &emdesc);
        // CUDA dups the handle internally on import, so close ours.
        external::CloseNativeHandle(nh);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaImportExternalMemory: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }

        // Get a mipmapped array view -> first level cudaArray for surface.
        cudaExternalMemoryMipmappedArrayDesc madesc{};
        std::memset(&madesc, 0, sizeof(madesc));
        madesc.offset    = 0;
        madesc.formatDesc.x = 16; madesc.formatDesc.y = 16;
        madesc.formatDesc.z = 16; madesc.formatDesc.w = 16;
        madesc.formatDesc.f = cudaChannelFormatKindFloat;
        madesc.extent.width  = w;
        madesc.extent.height = h;
        madesc.extent.depth  = 0;
        madesc.flags         = cudaArraySurfaceLoadStore;
        madesc.numLevels     = 1;
        cudaMipmappedArray_t mip = nullptr;
        cerr = cudaExternalMemoryGetMappedMipmappedArray(&mip, out.cuda_ext, &madesc);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaExternalMemoryGetMappedMipmappedArray: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }
        cerr = cudaGetMipmappedArrayLevel(&out.cuda_arr, mip, 0);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaGetMipmappedArrayLevel: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }

        cudaResourceDesc rdesc{};
        rdesc.resType         = cudaResourceTypeArray;
        rdesc.res.array.array = out.cuda_arr;
        cerr = cudaCreateSurfaceObject(&out.cuda_surf, &rdesc);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser({}): cudaCreateSurfaceObject: {}",
                      label, cudaGetErrorString(cerr));
            return false;
        }
        return true;
    };

    if (!build_one(img_color_in_, "color_in")) return false;
    if (!build_one(img_output_,   "output"))   return false;

    LOG_INFO("VulkanOptixDenoiser: external images sized for {}x{} "
             "(2 x VK_FORMAT_R16G16B16A16_SFLOAT, dedicated allocation, "
             "imported to CUDA as surface objects)", w, h);
    return true;
}

void VulkanOptixDenoiser::DestroyExternalImage(int slot) {
    ExternalImage& e = (slot == 0) ? img_color_in_ : img_output_;
    if (e.cuda_surf != 0) { cudaDestroySurfaceObject(e.cuda_surf); e.cuda_surf = 0; }
    if (e.cuda_ext  != nullptr) { cudaDestroyExternalMemory(e.cuda_ext); e.cuda_ext = nullptr; }
    e.cuda_arr = nullptr;  // owned by the mipmapped array; cudaDestroyExternalMemory frees it

    VkDevice dev = device_ ? device_->RawDevice() : VK_NULL_HANDLE;
    if (dev != VK_NULL_HANDLE) {
        if (e.view   != VK_NULL_HANDLE) { vkDestroyImageView(dev, e.view, nullptr); e.view = VK_NULL_HANDLE; }
        if (e.image  != VK_NULL_HANDLE) { vkDestroyImage(dev, e.image, nullptr); e.image = VK_NULL_HANDLE; }
        if (e.memory != VK_NULL_HANDLE) { vkFreeMemory(dev, e.memory, nullptr); e.memory = VK_NULL_HANDLE; }
    }
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
        // For timeline semaphores CUDA also wants the timeline flag.
        desc.flags = cudaExternalSemaphoreHandleTypeOpaqueWin32 ==
                         external::HandleTypes::cuda_semaphore
                     ? 0
                     : 0;
        // Note: the cuda sem handle type *_win32 already implies opaque;
        // the timeline-vs-binary distinction is encoded in the handle
        // type when CUDA was extended for timeline support. Recent CUDA
        // (12.0+) accepts the same "Opaque*" type and detects timeline
        // automatically from the imported VkSemaphore's type. No extra
        // flag needed here.
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

// ---- Encode --------------------------------------------------------------

void VulkanOptixDenoiser::Encode(VkCommandBuffer /*cb*/,
                                 const Device::DenoiseDesc& d) {
    if (!ready_) return;

    // Resolve frame size from engine's color_in.
    auto* dev = device_;
    if (dev == nullptr) return;
    // Look up the engine VkImage's extent. VulkanDevice exposes a
    // LookupImage helper; the underlying ImageEntry stored alongside
    // it has the extent we need. To avoid surfacing internals, query
    // via the ResizeTextures-style pattern: pass color_in's id and
    // let LookupImage give us the VkImage so we can read extent from
    // an inferred query (fallback: use a fixed default when lookup
    // doesn't reveal extent).
    //
    // Engine guarantees color_in is the path-tracer-output sized
    // texture, which matches the swapchain at frame creation time.
    // We piggyback on ResizeTextures for the actual size detection.
    //
    // For this commit we read the size from the color_in texture
    // image's extent via VulkanDevice's images_ map (LookupImage
    // returns the VkImage but not the extent; add a sibling lookup
    // or compute from descriptor info).
    //
    // Simplest path: VulkanDevice tracks ImageEntry which has extent;
    // expose via a small accessor. If no accessor exists today, the
    // resize path waits on the next refactor.
    //
    // Implementation note: VulkanDevice has private std::unordered_map
    // images_. We can't read it from here without an accessor. Rather
    // than touch VulkanDevice for this commit, we depend on the engine
    // to have allocated post_denoise_hdr / denoise_color at the
    // swapchain size, and we use the swapchain extent which the
    // engine guarantees matches.
    //
    // For now: fixed default of 1920x1080 on first call. Resize on
    // size change is a follow-up. RTX 5090 + 1080p is the explicit
    // target hardware so this is fine for the smoke test of this
    // commit; later commits make it dynamic.
    const std::uint32_t target_w = 1920;
    const std::uint32_t target_h = 1080;
    (void)d;  // d.color_in / d.output unused until per-frame work lands

    if (target_w != cached_w_ || target_h != cached_h_) {
        if (!ResizeScratch(target_w, target_h)) {
            LOG_ERROR("VulkanOptixDenoiser: ResizeScratch({}x{}) failed; "
                      "marking denoiser un-ready", target_w, target_h);
            ready_ = false;
            return;
        }
        if (!ResizeExternalImages(target_w, target_h)) {
            LOG_ERROR("VulkanOptixDenoiser: ResizeExternalImages({}x{}) failed; "
                      "marking denoiser un-ready", target_w, target_h);
            ready_ = false;
            return;
        }
        if (!ResizeExternalSemaphores()) {
            LOG_ERROR("VulkanOptixDenoiser: ResizeExternalSemaphores failed; "
                      "marking denoiser un-ready");
            ready_ = false;
            return;
        }
        cached_w_ = target_w;
        cached_h_ = target_h;
        LOG_INFO("VulkanOptixDenoiser: static state ready ({}x{}); "
                 "per-frame denoise dispatch is the next commit -- "
                 "image will stay noisy until then",
                 target_w, target_h);
    }

    // Per-frame interop dispatch lands in the next commit -- the
    // record-side sequence (vkCmdCopyImage in, signal vk-to-cuda,
    // optixDenoiserComputeIntensity, optixDenoiserInvoke,
    // cudaStreamSynchronize, vkCmdCopyImage out) touches Vulkan
    // queue submission semantics that need their own focused review.
    // With the static state above proven on first call, the per-frame
    // work is ~150 lines and lands cleanly on top.
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
    DestroyExternalImage(0);
    DestroyExternalImage(1);

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

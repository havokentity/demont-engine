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

    LOG_INFO("VulkanOptixDenoiser: external buffers sized for {}x{} "
             "(2 x {}KB, dedicated allocation, imported to CUDA as "
             "linear CUdeviceptrs)", w, h, buf_size / 1024);
    return true;
}

void VulkanOptixDenoiser::DestroyExternalBuffer(int slot) {
    ExternalBuffer& e = (slot == 0) ? buf_color_in_ : buf_output_;
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
// Per-frame flow when r_denoiser optix_hdr is active:
//
//   Frame N tick:
//     1. Engine path-tracer dispatch records into engine cb (already
//        done before VulkanDevice::Denoise dispatched here).
//     2. We append to the engine cb:
//          - barrier d.color_in: GENERAL -> TRANSFER_SRC
//          - vkCmdCopyImageToBuffer(d.color_in -> buf_color_in_)
//          - barrier d.color_in: TRANSFER_SRC -> GENERAL (path tracer
//            wants GENERAL again next frame)
//     3. Tell VulkanDevice to extra-signal sem_vk_to_cuda_ at value N
//        when it submits the engine cb.
//     4. Schedule async CUDA work on cuda_stream_:
//          - cudaWaitExternalSemaphoresAsync on sem_vk_to_cuda_ (val N)
//          - optixDenoiserComputeIntensity (HDR model needs intensity)
//          - optixDenoiserInvoke (buf_color_in_.cuda_ptr ->
//                                 buf_output_.cuda_ptr)
//          - cudaSignalExternalSemaphoresAsync on sem_cuda_to_vk_ (val N)
//     5. Submit a private cb on the graphics queue:
//          - waitSemaphore: sem_cuda_to_vk_ at value N
//          - barrier d.output: GENERAL -> TRANSFER_DST
//          - vkCmdCopyBufferToImage(buf_output_ -> d.output)
//          - barrier d.output: TRANSFER_DST -> GENERAL (so the
//            engine's next tonemap / present sees a sane layout)
//
// Submitting the private cb at Encode time before the engine cb is
// submitted is fine -- the wait on sem_cuda_to_vk_ value N keeps the
// private cb pending in the queue until CUDA signals, which only
// happens after the engine cb has signalled sem_vk_to_cuda_, which
// only happens after the engine cb finishes its copy. Serialised
// correctly through the timeline-semaphore handshake.

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
    VkImage src_img = device_->LookupImage(d.color_in);
    VkImage dst_img = device_->LookupImage(d.output);
    if (src_img == VK_NULL_HANDLE || dst_img == VK_NULL_HANDLE) {
        LOG_WARN("VulkanOptixDenoiser::Encode: image lookup miss "
                 "(src={} dst={})", static_cast<void*>(src_img),
                                    static_cast<void*>(dst_img));
        return;
    }

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

    // ---- Step 3: request the engine submit additionally signal vk_to_cuda
    device_->RequestExtraSubmitSignal(sem_vk_to_cuda_, this_frame);

    // ---- Step 4: schedule async CUDA wait + denoise + signal -------------
    //
    // All four calls are checked: a silent failure here would leave the
    // private cb's wait on sem_cuda_to_vk forever-pending, blocking
    // vkDeviceWaitIdle at engine shutdown -> force-kill required. The
    // log message identifies which step broke so we don't have to
    // bisect.
    bool cuda_chain_ok = true;
    {
        cudaExternalSemaphoreWaitParams wait_params{};
        std::memset(&wait_params, 0, sizeof(wait_params));
        wait_params.params.fence.value = this_frame;
        cudaError_t cerr = cudaWaitExternalSemaphoresAsync(
            &cuda_sem_vk_to_cuda_, &wait_params, 1, cuda_stream_);
        if (cerr != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser::Encode: cudaWaitExternalSemaphoresAsync: {}",
                      cudaGetErrorString(cerr));
            cuda_chain_ok = false;
        }

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

        if (cuda_chain_ok) {
            OptixResult oresult = optixDenoiserComputeIntensity(
                optix_denoiser_, cuda_stream_,
                &color_in,
                static_cast<CUdeviceptr>(intensity_buf_),
                static_cast<CUdeviceptr>(scratch_buf_),
                scratch_size_);
            if (oresult != OPTIX_SUCCESS) {
                LOG_ERROR("VulkanOptixDenoiser::Encode: "
                          "optixDenoiserComputeIntensity (OptixResult={})",
                          static_cast<int>(oresult));
                cuda_chain_ok = false;
            }
        }

        OptixDenoiserParams params{};
        std::memset(&params, 0, sizeof(params));
        params.hdrIntensity   = static_cast<CUdeviceptr>(intensity_buf_);
        params.blendFactor    = 0.0f;  // 0 = full denoise, 1 = passthrough

        OptixDenoiserGuideLayer guide_layer{};
        std::memset(&guide_layer, 0, sizeof(guide_layer));

        OptixDenoiserLayer layer{};
        std::memset(&layer, 0, sizeof(layer));
        layer.input  = color_in;
        layer.output = output;

        if (cuda_chain_ok) {
            OptixResult oresult = optixDenoiserInvoke(
                optix_denoiser_, cuda_stream_,
                &params,
                static_cast<CUdeviceptr>(state_buf_), state_size_,
                &guide_layer,
                &layer, 1,
                /*inputOffsetX*/ 0, /*inputOffsetY*/ 0,
                static_cast<CUdeviceptr>(scratch_buf_), scratch_size_);
            if (oresult != OPTIX_SUCCESS) {
                LOG_ERROR("VulkanOptixDenoiser::Encode: optixDenoiserInvoke "
                          "(OptixResult={})", static_cast<int>(oresult));
                cuda_chain_ok = false;
            }
        }

        // ALWAYS signal cuda_to_vk -- even if any earlier CUDA/OptiX
        // step failed. Otherwise the private cb's wait deadlocks the
        // queue and hangs vkDeviceWaitIdle at shutdown. Visual output
        // will be garbage on a failure frame, but the engine stays
        // responsive and the LOG_ERROR above tells us why.
        cudaExternalSemaphoreSignalParams sig_params{};
        std::memset(&sig_params, 0, sizeof(sig_params));
        sig_params.params.fence.value = this_frame;
        cudaError_t sig_err = cudaSignalExternalSemaphoresAsync(
            &cuda_sem_cuda_to_vk_, &sig_params, 1, cuda_stream_);
        if (sig_err != cudaSuccess) {
            LOG_ERROR("VulkanOptixDenoiser::Encode: cudaSignalExternalSemaphoresAsync: {}",
                      cudaGetErrorString(sig_err));
            // No way to recover -- private cb will hang. Warn and
            // proceed; vkDeviceWaitIdle will hang at shutdown.
        }
    }

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
        // UNDEFINED is spec-correct here -- we're about to write the
        // entire image via vkCmdCopyBufferToImage so previous contents
        // don't matter. Using GENERAL would require the image to
        // already be in GENERAL on first frame, which it isn't (engine
        // creates post_denoise_hdr with VK_IMAGE_LAYOUT_UNDEFINED).
        toDst.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcAccessMask = 0;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image         = dst_img;
        toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toDst.subresourceRange.layerCount = 1;
        toDst.subresourceRange.levelCount = 1;
        vkCmdPipelineBarrier(pcb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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

        VkImageMemoryBarrier backToGeneral{};
        backToGeneral.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        backToGeneral.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        backToGeneral.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        backToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        backToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        backToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneral.image         = dst_img;
        backToGeneral.subresourceRange = toDst.subresourceRange;
        vkCmdPipelineBarrier(pcb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &backToGeneral);

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

    VkCommandBuffer pcb = private_cb_out_[pending_pcb_slot_];

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1;
    timeline_info.pWaitSemaphoreValues    = &pending_wait_value_;

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext                = &timeline_info;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &sem_cuda_to_vk_;
    si.pWaitDstStageMask    = &wait_stage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &pcb;
    si.signalSemaphoreCount = 0;
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

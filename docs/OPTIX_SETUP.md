# OptiX Denoiser Setup

The OptiX denoiser path (`r_denoiser optix_hdr` / `optix_hdr_aov`) requires
a CUDA Toolkit, an OptiX SDK, and an NVIDIA driver whose bundled
`nvoptix.dll` (or `libnvoptix.so` on Linux) supports the ABI of the
SDK headers we compile against.

## Default setup

Tested working on:

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5090 |
| Driver | 596.36 (Windows) |
| CUDA Toolkit | 13.1.80 |
| OptiX SDK | 9.1.0 |

Install:

1. **CUDA Toolkit 12.0+** — https://developer.nvidia.com/cuda-downloads.
   CMake's `find_package(CUDAToolkit)` picks it up automatically.
2. **OptiX SDK 9.x** — https://developer.nvidia.com/designworks/optix/download
   (free NVIDIA Developer account required).
   - Default install path on Windows:
     `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\`.
   - The CMake `find_path()` glob walks
     `C:/ProgramData/NVIDIA Corporation/OptiX SDK *` so any 8.x / 9.x
     install picks up. Override with the `OPTIX_INSTALL_DIR` env var
     if installed elsewhere.
3. **NVIDIA Driver R580+** if using OptiX 9.1 SDK (R555+ for OptiX 9.0).
   The driver ships `nvoptix.dll` whose ABI must match the SDK headers.

## Troubleshooting

### `optixInit() failed (OptixResult=7801)` at runtime

`7801` = `OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND`. The driver's bundled
OptiX runtime is older than the SDK ABI we're compiling against.

Two fixes (pick one):

- **Update NVIDIA driver** to one that ships an OptiX runtime matching
  your SDK version (R580+ for OptiX 9.1, R555+ for OptiX 9.0). Most
  driver releases bump OptiX in step.
- **Or downgrade SDK to OptiX 9.0** — install OptiX 9.0.0 (or 9.0.2)
  from the NVIDIA developer downloads page alongside your existing
  install, then either uninstall 9.1 or set
  `OPTIX_INSTALL_DIR=...\OptiX SDK 9.0.0` in the shell that runs
  `cmake`. The functional difference between 9.0 and 9.1 is minor
  (slightly improved NN denoiser model in 9.1, no API removals).
  9.0 trades a small quality difference for compatibility with a
  wider range of drivers.

To see which `nvoptix.dll` version your active driver shipped:

```powershell
Get-ChildItem -Path 'C:/Windows' -Recurse -Filter 'nvoptix*.dll' -ErrorAction SilentlyContinue |
    Select-Object @{n='Version';e={$_.VersionInfo.FileVersion}}, FullName
```

### `vkGetMemoryWin32HandleKHR unresolved`

The Vulkan device wasn't created with `VK_KHR_external_memory_win32`
enabled. The engine enables this automatically when `PT_ENABLE_OPTIX`
is on (see `src/rhi_vulkan/VulkanDevice.cpp`); if you see this error
on a non-NVIDIA Vulkan driver, OptiX is unavailable on that hardware
anyway -- pick `r_denoiser svgf_atrous` or `off`.

### `cudaImportExternalSemaphore: invalid argument`

Usually means the Vulkan 1.2 `timelineSemaphore` feature wasn't
enabled, so `vkCreateSemaphore` silently fell back to a binary
semaphore that CUDA then can't import. The engine enables this
feature when `PT_ENABLE_OPTIX` is on. If you still hit this on a
GPU that should support it, check that your driver isn't from
before R535 (the cutoff for stable timeline-semaphore + CUDA
interop on NVIDIA).

### Build configures with `PT_OPTIX_ACTIVE OFF`

CMake couldn't find one of the SDKs. The configure pass prints a
`STATUS` line saying which is missing -- check the cmake output
for `PT_ENABLE_OPTIX:` lines. Common causes:

- `OPTIX_INSTALL_DIR` env var wasn't visible to the cmake shell
  (system-set vars only propagate to processes started after the
  change -- close + reopen terminals after editing System Variables).
- OptiX SDK installed in a non-standard path. Set `OPTIX_INSTALL_DIR`
  manually or pass `-DOPTIX_INCLUDE_DIR=<path>/include` to cmake.
- CUDA Toolkit 12.0+ not on PATH. `nvcc --version` should print
  `release 12.0` or higher.

## Linux

Path is structured (see `src/rhi_vulkan/ExternalHandles.h` for the
FD-handle branch), but **untested as of v0.3.1**. Will be validated on
a real Linux box when one is available. Expected adjustments are
small -- typically a missing `#include` or a CMake hint, no
architectural changes. If you build on Linux and hit issues, please
file an issue with the cmake configure output + a snippet of the
build error and we'll fix.

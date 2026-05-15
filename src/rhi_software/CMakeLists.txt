# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi_software STATIC
    SoftwareDevice.cpp
    SoftwareTracer.cpp
)

if(WIN32)
    # Win32-only: Vulkan-blit present path. Lets the Software backend
    # keep displaying after the Vulkan backend ever rendered to the
    # window -- DXGI flip-model permanently locks the HWND out of
    # GDI presentation, so GDI alone can't recover from a vulkan ->
    # software switch. The Vulkan-blit path keeps the window in
    # flip-model presentation throughout. See SoftwareVulkanPresent.h
    # for the full design rationale + spec receipts. Driven by the
    # r_software_blit cvar (vulkan default, gdi opt-in).
    target_sources(pt_rhi_software PRIVATE SoftwareVulkanPresent.cpp)
endif()

target_include_directories(pt_rhi_software PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_software
    PUBLIC pt_core
    PUBLIC pt_app_window     # for pt_window_native_cocoa / pt_window_native_win32 / pt_metal_attach_layer
    PRIVATE pt_console       # r_software_blit cvar lookup at Init time (Win32 only at the use site, but cheap to link unconditionally)
    PRIVATE embree           # CPU BVH + ray-triangle intersection (PathTrace.slang's RayQuery equivalent on the CPU side)
    PRIVATE glm::glm         # vec/mat math for the CPU kernel
)

if(APPLE)
    # Software backend uses Metal for the present blit. The metal-cpp
    # PRIVATE_IMPLEMENTATION lives in pt_rhi_metal; we just consume the
    # headers and the same Metal frameworks.
    target_link_libraries(pt_rhi_software PUBLIC
        metal_cpp
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
elseif(WIN32)
    # Software backend's Windows present paths:
    #   - GDI (legacy / opt-in): SetDIBitsToDevice into the HWND's HDC.
    #     gdi32 + user32 are implicitly linked by the MSVC / clang-cl
    #     default lib set; listing them explicitly documents the
    #     dependency for anyone reading the build.
    #   - Vulkan-blit (default): SoftwareVulkanPresent owns its own
    #     VkInstance / VkDevice / VkSwapchainKHR, mutually exclusive
    #     with VulkanDevice's instance (only one RHI backend runs at
    #     a time so there's no conflict at the API level). glfw is
    #     pulled in for the GLFW_INCLUDE_VULKAN header glue, matching
    #     the pattern in pt_rhi_vulkan/CMakeLists.txt.
    target_link_libraries(pt_rhi_software PRIVATE
        gdi32 user32
        Vulkan::Vulkan
        glfw)
endif()

pt_set_compiler_options(pt_rhi_software)

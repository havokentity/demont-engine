# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_rhi_software STATIC
    SoftwareDevice.cpp
    SoftwareTracer.cpp
)

target_include_directories(pt_rhi_software PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_software
    PUBLIC pt_core
    PUBLIC pt_app_window     # for pt_metal_attach_layer / pt_window_native_cocoa
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
endif()

pt_set_compiler_options(pt_rhi_software)

add_library(pt_rhi_vulkan STATIC
    VulkanDevice.cpp
)

target_include_directories(pt_rhi_vulkan PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_vulkan
    PUBLIC pt_core
    PUBLIC pt_app_window           # GLFW + (later) Cocoa bridge
    PUBLIC Vulkan::Vulkan
    PUBLIC glfw                    # for glfwCreateWindowSurface
)

# Single unified shader: dynamic analytic primitives + triangle meshes.
include(${CMAKE_SOURCE_DIR}/cmake/Slang.cmake)
pt_compile_slang(
    TARGET pt_rhi_vulkan SOURCE ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE compute ENTRY main TARGETS spirv)

pt_set_compiler_options(pt_rhi_vulkan)

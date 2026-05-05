add_library(pt_rhi_metal STATIC
    MetalDevice.cpp
)

target_include_directories(pt_rhi_metal PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_metal
    PUBLIC pt_core
    PUBLIC pt_app_window     # for pt_metal_attach_layer + pt_window_native_cocoa
    PUBLIC metal_cpp
)

if(APPLE)
    target_link_libraries(pt_rhi_metal PUBLIC
        "-framework Metal"
        "-framework MetalKit"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

# Compile the clear + scene shaders (Slang -> MSL source) and embed.
include(${CMAKE_SOURCE_DIR}/cmake/Slang.cmake)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/Clear.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/Scene.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/PathTrace.slang
    STAGE   compute
    ENTRY   main
    TARGETS metal
)

pt_set_compiler_options(pt_rhi_metal)

# metal-cpp uses Objective-C runtime under the hood; suppress -Wpedantic
# warnings about extensions in the metal-cpp headers, and don't apply our
# -fno-rtti to the impl TU since metal-cpp objc runtime needs basic typeid
# in some configs.
target_compile_options(pt_rhi_metal PRIVATE
    -Wno-deprecated-declarations
    -Wno-pedantic
)

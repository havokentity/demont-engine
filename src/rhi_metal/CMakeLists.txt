add_library(pt_rhi_metal STATIC
    MetalDevice.cpp
    MetalAttach.mm
)

target_include_directories(pt_rhi_metal PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_metal
    PUBLIC pt_core
    PUBLIC metal_cpp
)

if(APPLE)
    target_link_libraries(pt_rhi_metal PUBLIC
        "-framework Metal"
        "-framework MetalKit"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework AppKit"
        "-framework CoreGraphics"
    )
    set_source_files_properties(MetalAttach.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# Compile the clear shader (Slang -> MSL source) and embed.
include(${CMAKE_SOURCE_DIR}/cmake/Slang.cmake)
pt_compile_slang(
    TARGET  pt_rhi_metal
    SOURCE  ${CMAKE_SOURCE_DIR}/shaders/Clear.slang
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

add_library(pt_rhi STATIC
    Device.cpp
)

target_include_directories(pt_rhi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_rhi
    PUBLIC pt_core
    PUBLIC pt_rhi_software
)

pt_set_compiler_options(pt_rhi)

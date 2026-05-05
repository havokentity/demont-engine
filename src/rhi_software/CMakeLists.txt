add_library(pt_rhi_software STATIC
    SoftwareDevice.cpp
)

target_include_directories(pt_rhi_software PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_rhi_software
    PUBLIC pt_core
    PUBLIC glfw
)

if(APPLE)
    target_link_libraries(pt_rhi_software PUBLIC
        "-framework OpenGL"
    )
endif()

pt_set_compiler_options(pt_rhi_software)

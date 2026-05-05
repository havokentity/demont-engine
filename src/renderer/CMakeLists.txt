add_library(pt_renderer INTERFACE)
target_include_directories(pt_renderer INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_renderer INTERFACE pt_core)

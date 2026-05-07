# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_engine STATIC
    Engine.cpp
)

target_include_directories(pt_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_engine
    PUBLIC pt_core
    PUBLIC pt_console
    PUBLIC pt_app_window
    PUBLIC pt_rhi
    PUBLIC pt_renderer
    PRIVATE tomlplusplus::tomlplusplus
)

pt_set_compiler_options(pt_engine)

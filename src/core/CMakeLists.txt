# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_core STATIC
    AssetPath.cpp
    Diag.cpp
    Log.cpp
    Memory/Memory.cpp
    Memory/FrameArena.cpp
    Jobs/JobSystem.cpp
    Hardware/HardwareInfo.cpp
)

target_include_directories(pt_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_core
    PUBLIC  fmt::fmt
    PUBLIC  glm::glm
    PUBLIC  mimalloc-static
    PUBLIC  enkiTS
)

if(PT_ENABLE_TRACY)
    target_link_libraries(pt_core PUBLIC TracyClient)
    target_compile_definitions(pt_core PUBLIC TRACY_ENABLE=1)
endif()

pt_set_compiler_options(pt_core)

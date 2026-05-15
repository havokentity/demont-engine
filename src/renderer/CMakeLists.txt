# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_renderer STATIC
    AnalyticBvh.cpp
    Astronomy.cpp
    BscCatalog.cpp
    Csg/CsgScene.cpp
    HdrImage.cpp
    MoonTexture.cpp
)
target_include_directories(pt_renderer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_renderer
    PUBLIC  pt_core
    PRIVATE manifold
)

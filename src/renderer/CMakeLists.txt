add_library(pt_renderer STATIC
    Astronomy.cpp
    BscCatalog.cpp
    Csg/CsgScene.cpp
    HdrImage.cpp
)
target_include_directories(pt_renderer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pt_renderer
    PUBLIC  pt_core
    PRIVATE manifold
)

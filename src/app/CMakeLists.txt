# --- Window library (separate from the executable so engine can link it) ---
add_library(pt_app_window STATIC
    Window.cpp
    Window_Cocoa.mm
)

target_include_directories(pt_app_window PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_app_window
    PUBLIC  pt_core
    PUBLIC  glfw
)

if(APPLE)
    target_link_libraries(pt_app_window PUBLIC
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation"
        "-framework QuartzCore"
    )
    # The Cocoa bridge is Objective-C++; do NOT apply -fno-rtti / -fno-exceptions
    # to it (clang complains about rtti for ObjC selectors). Set per-source.
    set_source_files_properties(Window_Cocoa.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

pt_set_compiler_options(pt_app_window)

# --- Executable ------------------------------------------------------------
add_executable(pathtracer
    main.cpp
)

target_link_libraries(pathtracer PRIVATE
    pt_core
    pt_console
    pt_engine
    pt_app_window
    pt_rhi
)

pt_set_compiler_options(pathtracer)

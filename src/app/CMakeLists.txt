# --- Window library (separate from the executable so engine can link it) ---
add_library(pt_app_window STATIC
    Window.cpp
    Window_Cocoa.mm
    MetalLayerAttach.mm
    ConsoleOverlay.mm
)

target_include_directories(pt_app_window PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_app_window
    PUBLIC  pt_core
    PUBLIC  pt_console
    PUBLIC  glfw
)

if(APPLE)
    target_link_libraries(pt_app_window PUBLIC
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation"
        "-framework QuartzCore"
        "-framework CoreText"
        "-framework CoreGraphics"
    )
    # Objective-C++ files use ARC. Do not apply -fno-rtti to them.
    set_source_files_properties(Window_Cocoa.mm     PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(MetalLayerAttach.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(ConsoleOverlay.mm   PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

pt_set_compiler_options(pt_app_window)

# --- Executable ------------------------------------------------------------
add_executable(demont
    main.cpp
)

target_link_libraries(demont PRIVATE
    pt_core
    pt_console
    pt_engine
    pt_app_window
    pt_rhi
)

pt_set_compiler_options(demont)

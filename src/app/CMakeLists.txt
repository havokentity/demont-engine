add_executable(pathtracer
    main.cpp
)

target_link_libraries(pathtracer PRIVATE pt_core)

pt_set_compiler_options(pathtracer)

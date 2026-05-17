# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# pt_render_one_frame: thin harness wrapper that translates the
# golden-image regression matrix (#45) test-facing flags into the
# engine's smoke-test CLI and spawns the demont binary. Deliberately
# not linked to any pt_* engine library or to Vulkan / Metal / OptiX
# / stb -- the wrapper only forks the engine child and writes a tiny
# temp cfg script. Builds in milliseconds on any host.

add_executable(pt_render_one_frame main.cpp)

# Apply the project's strict warning baseline like the other tools/ dirs.
pt_set_compiler_options(pt_render_one_frame)

# Make ctest cells able to resolve $<TARGET_FILE:pt_render_one_frame>
# without depending on the build-tree directory layout. They also
# depend on demont being present, so register the dependency so a
# clean build of pt_render_one_frame implicitly builds demont -- if
# someone runs `cmake --build . --target pt_render_one_frame`, the
# matrix cells produced from it actually work without a second pass.
if(TARGET demont)
    add_dependencies(pt_render_one_frame demont)
endif()

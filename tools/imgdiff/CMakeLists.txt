# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# imgdiff: standalone PNG comparison CLI used by the golden-image
# regression matrix (#45). Issue #69.
#
# Deliberately not linked to any pt_* engine library -- the tool needs
# to build and run without dragging in Vulkan / Metal / Embree / Tracy.
# Only deps are the vendored stb single-headers in third_party/stb/.

add_executable(imgdiff
    main.cpp
    stb_impl.cpp
)

# stb headers live at third_party/stb/{stb_image.h,stb_image_write.h};
# stb_impl.cpp does the one-TU `#define STB_*_IMPLEMENTATION` so the
# include path is all that's needed.
target_include_directories(imgdiff PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# stb's implementations emit a flood of -Wpedantic / -Wsign-conversion /
# C4244 (and friends) warnings that we can't fix without forking.
# Match the policy used elsewhere in the project (Embree, civetweb,
# Tracy): silence at file scope only for the stb-impl TU. main.cpp
# keeps the project's normal strict-warning policy so any future
# regression there still trips the build.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(stb_impl.cpp PROPERTIES
        COMPILE_FLAGS "/w")
else()
    set_source_files_properties(stb_impl.cpp PROPERTIES
        COMPILE_FLAGS "-w")
endif()

# pt_set_compiler_options applies the project's C++23 + no-exceptions +
# no-rtti baseline + -Wall/-Wextra/-Wpedantic (or /W4 /permissive- on
# MSVC). On stb_impl.cpp the per-file /w / -w above takes precedence
# for warnings; on main.cpp the strict warnings remain in force.
# pt_set_compiler_options is defined in cmake/CompilerWarnings.cmake.
pt_set_compiler_options(imgdiff)

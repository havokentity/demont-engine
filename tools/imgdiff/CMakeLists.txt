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
)

# stb headers live at third_party/stb/{stb_image.h,stb_image_write.h};
# main.cpp does the one-TU `#define STB_IMAGE_IMPLEMENTATION` so the
# include path is all that's needed -- no separate translation unit.
target_include_directories(imgdiff PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# stb's implementations issue a flood of -Wpedantic / -Wsign-conversion /
# C4244 (and friends) warnings that we can't fix without forking. Match
# the policy used elsewhere in the project (Embree, civetweb, Tracy):
# silence at file scope only for the TU that pulls in the stb impl.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(main.cpp PROPERTIES
        COMPILE_FLAGS "/w")
else()
    set_source_files_properties(main.cpp PROPERTIES
        COMPILE_FLAGS "-w")
endif()

# Still apply the standard C++23 flag + no-exceptions / no-rtti policy
# to the target -- the per-file COMPILE_FLAGS only silences warnings,
# the language baseline still comes from pt_set_compiler_options.
# pt_set_compiler_options is defined in cmake/CompilerWarnings.cmake
# and is already included from the root.
pt_set_compiler_options(imgdiff)

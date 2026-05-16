// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Dedicated TU for the stb_image_write implementation. Confined here so
// the warning flood from stb's macro-heavy single-header style (~30
// /W4 warnings on MSVC, ~50 -Wall on GCC/Clang) doesn't poison the
// rest of `pt_capture_encoder` / `pt_engine`. The owning CMakeLists
// applies per-file `/w` (MSVC) / `-w` (Clang/GCC) only to this TU --
// see src/engine/CMakeLists.txt.
//
// Same pattern PR #92 (imgdiff) established at
// tools/imgdiff/stb_impl.cpp.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb/stb_image_write.h"

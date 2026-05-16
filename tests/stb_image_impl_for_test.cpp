// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Test-only stb_image implementation TU. Used by tests/capture_format_test.cpp
// to LOAD back a PNG written by `EncodeAndWritePng` and verify pixel-exact
// preservation through the ACES + sRGB pipeline.
//
// Confined to its own TU so stb's warning flood is silenced by per-file
// `/w` (MSVC) / `-w` (Clang/GCC); see tests/CMakeLists.txt. Same pattern
// the engine uses at src/engine/stb_impl.cpp.

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

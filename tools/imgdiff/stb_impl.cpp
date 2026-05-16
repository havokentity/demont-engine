// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// One-TU home for the stb_image / stb_image_write implementations.
// Kept separate from main.cpp so we can silence stb's warning flood at
// file scope (matching the repo's third-party warning policy in
// cmake/Dependencies.cmake) without also weakening warnings on the
// CLI driver's own code in main.cpp.
//
// No public symbols are added here -- stb's functions reach main.cpp
// via the headers' regular declarations (which main.cpp #includes).

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

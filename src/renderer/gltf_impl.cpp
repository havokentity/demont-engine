// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// Dedicated TU for cgltf + stb_image single-header implementations.
// Confined here so the warning flood from these macro-heavy vendored
// headers (~100+ /W4 warnings on MSVC, ~100+ -Wall on GCC/Clang)
// doesn't poison the rest of pt_renderer. The owning CMakeLists
// applies per-file `/w` (MSVC) / `-w` (Clang/GCC) only to this TU --
// see src/renderer/CMakeLists.txt.
//
// Sister files using the same pattern:
//   - src/engine/stb_impl.cpp         (stb_image_write)
//   - tools/imgdiff/stb_impl.cpp      (stb_image / stb_image_write)
//   - tests/stb_image_impl_for_test.cpp (stb_image)
//
// This TU intentionally contains no project logic so the warning
// suppression has no scope to mask real bugs. GltfImporter.cpp keeps
// the strict project warning policy.

// cgltf is the upstream single-header glTF 2.0 parser (MIT, jkuhlmann).
// Vendored at third_party/cgltf/cgltf.h, see issue #79 for rationale
// (single-header + no transitive deps + actively-maintained + the
// idiomatic choice in the Khronos ecosystem). This TU is the SOLE
// definer of CGLTF_IMPLEMENTATION in the codebase -- if a future TU
// needs the parser, it should #include "GltfImporter.h" and call
// LoadGltf, not redefine the impl.
#define CGLTF_IMPLEMENTATION
#include "../../third_party/cgltf/cgltf.h"

// stb_image's read path is used by GltfImporter for the optional
// baseColorTexture. The rest of the engine uses stb_image_write
// (separate symbol table) from src/engine/stb_impl.cpp; CGLTF and
// STB share no symbols so it's safe to define both impls in this TU.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR        // We use Radiance.hdr via our own loader.
#define STBI_NO_LINEAR
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_GIF
#define STBI_NO_TGA
#include "../../third_party/stb/stb_image.h"

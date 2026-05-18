# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte

# --- pt_capture_encoder ---------------------------------------------------
# Pixel encoders for FrameCapture: ACES + sRGB tonemap helper, PPM writer,
# PNG writer (via stb_image_write). Split out of pt_engine so:
#   (1) the stb_image_write implementation is confined to ONE TU
#       (`stb_impl.cpp`) with per-file `/w` / `-w` suppressing stb's
#       /W4 warning flood, without touching the rest of pt_engine.
#   (2) unit tests can link the encoder in isolation -- no RHI, no
#       renderer, no Vulkan SDK -- so `tests/capture_format_test.cpp`
#       stays a fast, deterministic, cross-platform doctest binary.
#
# Same pattern PR #92 (imgdiff) established at tools/imgdiff/.
add_library(pt_capture_encoder STATIC
    CaptureEncoder.cpp
    stb_impl.cpp
)

target_include_directories(pt_capture_encoder PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_capture_encoder
    PUBLIC pt_core
)

pt_set_compiler_options(pt_capture_encoder)

# Per-file warning suppression on stb_impl.cpp. The project applies
# /W4 (MSVC) / -Wall -Wextra -Wpedantic (Clang/GCC) globally via
# pt_set_compiler_options above; cl.exe and gcc both honour the
# rightmost warning-level flag, so appending /w / -w via file-level
# COMPILE_OPTIONS overrides for this single TU. pt_engine and the
# rest of pt_capture_encoder keep their hard-warning policy.
if(MSVC)
    set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

# --- pt_engine ------------------------------------------------------------
add_library(pt_engine STATIC
    Engine.cpp
    FrameCapture.cpp
    LensFlare.cpp
)

target_include_directories(pt_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_engine
    PUBLIC  pt_core
    PUBLIC  pt_console
    PUBLIC  pt_app_window
    PUBLIC  pt_rhi
    PUBLIC  pt_renderer
    PUBLIC  pt_audio
    PUBLIC  pt_effects
    PRIVATE pt_capture_encoder
    PRIVATE tomlplusplus::tomlplusplus
)

pt_set_compiler_options(pt_engine)

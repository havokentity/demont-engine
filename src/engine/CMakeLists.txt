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
    PUBLIC  pt_physics
    PUBLIC  pt_destruction
    PRIVATE pt_capture_encoder
    PRIVATE tomlplusplus::tomlplusplus
)

# --- Megakernel feature gates (PR #164) -----------------------------------
# Mirror the shader-side PT_WATER_ENABLED gate into a C++ compile define so
# Engine.cpp can emit a single startup [warn] when a user has tweaked the
# r_water_* cvars in a build that compiled out the water BRDF.
if(PT_WATER_ENABLED)
    target_compile_definitions(pt_engine PRIVATE PT_WATER_ENABLED=1)
else()
    target_compile_definitions(pt_engine PRIVATE PT_WATER_ENABLED=0)
endif()
# --- end Megakernel feature gates -----------------------------------------

pt_set_compiler_options(pt_engine)

# --- SDF Phase 2 (#98) megakernel-perf gates: host-side mirror -----------
# The PT_SDF_PROCEDURAL_OPS / PT_SDF_AUTODIFF cache options (see
# cmake/Slang.cmake) gate the shader-side Phase 2 procedural + Dual3
# autodiff blocks. The engine needs to know whether the AD path was
# compiled in so we can emit a [warn] line at startup when
# r_sdf_normal_mode=0 (forward-AD) but this build doesn't ship it. We
# mirror both flags into C++ preprocessor land so the warning lives in
# the same translation unit as the cvar registration.
if(PT_SDF_PROCEDURAL_OPS)
    target_compile_definitions(pt_engine PRIVATE PT_SDF_PROCEDURAL_OPS=1)
else()
    target_compile_definitions(pt_engine PRIVATE PT_SDF_PROCEDURAL_OPS=0)
endif()
if(PT_SDF_AUTODIFF)
    target_compile_definitions(pt_engine PRIVATE PT_SDF_AUTODIFF=1)
else()
    target_compile_definitions(pt_engine PRIVATE PT_SDF_AUTODIFF=0)
endif()

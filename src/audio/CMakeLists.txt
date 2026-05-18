# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# pt_audio -- miniaudio-backed audio subsystem (issue #80, MVP).
#
# Single-header dep (third_party/miniaudio/miniaudio.h) is compiled
# inline inside AudioSystem.cpp via #define MINIAUDIO_IMPLEMENTATION.
# This is the same single-impl-TU pattern engine/stb_impl.cpp uses for
# stb_image_write.
add_library(pt_audio STATIC
    AudioSystem.cpp
)

target_include_directories(pt_audio PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_audio
    PUBLIC  pt_core
    PUBLIC  glm::glm
)

# Platform audio backends:
#   - macOS:  -framework CoreAudio / AudioToolbox / AudioUnit / CoreFoundation
#             (miniaudio picks CoreAudio at runtime via MA_BACKEND_COREAUDIO).
#   - Linux:  libdl (miniaudio loads ALSA / PulseAudio symbols dynamically).
#   - Win32:  WASAPI is header-only via <mmdeviceapi.h>; no extra libs needed.
#             (`ole32` is already linked by other engine TUs via Win32 default;
#             explicit add here keeps this CMakeLists self-contained.)
if(APPLE)
    target_link_libraries(pt_audio
        PUBLIC  "-framework CoreFoundation"
        PUBLIC  "-framework CoreAudio"
        PUBLIC  "-framework AudioToolbox"
        PUBLIC  "-framework AudioUnit"
    )
elseif(UNIX)
    target_link_libraries(pt_audio PUBLIC dl pthread m)
elseif(WIN32)
    target_link_libraries(pt_audio PUBLIC ole32)
endif()

pt_set_compiler_options(pt_audio)

# Per-file warning suppression on AudioSystem.cpp. miniaudio.h is a
# single-header library full of integer-conversion / unused-parameter
# warnings under -Wall -Wextra; engine/stb_impl.cpp uses the same
# pattern for stb_image_write.h. The project's PUBLIC compiler-warning
# policy is preserved for every other pt_audio TU (none exist today,
# but anything added later inherits the strict policy by default).
if(MSVC)
    set_source_files_properties(AudioSystem.cpp PROPERTIES COMPILE_OPTIONS "/w")
else()
    set_source_files_properties(AudioSystem.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

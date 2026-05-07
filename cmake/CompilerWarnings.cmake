# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Apply orthodox-C++23 compile/link flags to a single target.
# Per principles doc §08: no exceptions, no RTTI, hard warnings.
# ASan is opt-in via PT_ENABLE_SANITIZERS (mimalloc + ASan needs MI_TRACK_ASAN).
#
# Cross-platform: AppleClang/Clang/GCC use the canonical -flags;
# MSVC (and clang-cl in MSVC-driver mode) gets equivalent /flags.

function(pt_set_compiler_options target)
    target_compile_features(${target} PUBLIC cxx_std_23)

    if(MSVC)
        # /EHs-c-  : disable C++ exception handling (assume noexcept everywhere)
        # /GR-     : disable RTTI
        # /W4      : warning level 4 (close to -Wall -Wextra)
        # /permissive- : tighten conformance (closer to -Wpedantic)
        # /wd4100  : unused parameter (mirrors -Wno-unused-parameter)
        # /Zc:__cplusplus : report the real __cplusplus value (MSVC defaults to 199711L)
        target_compile_options(${target} PRIVATE
            /EHs-c-
            /GR-
            /W4
            /permissive-
            /Zc:__cplusplus
            /wd4100   # unused formal parameter
            /wd4324   # struct padded due to alignment
        )
        target_compile_definitions(${target} PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:/Od>
            $<$<CONFIG:Debug>:/Zi>
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            $<$<CONFIG:Release>:/DNDEBUG>
        )
    else()
        target_compile_options(${target} PRIVATE
            -fno-exceptions
            -fno-rtti
            -fvisibility=hidden
            -fvisibility-inlines-hidden
            -Wall -Wextra -Wpedantic
            -Werror=return-type
            -Wshadow
            -Wno-unknown-pragmas
            -Wno-unused-parameter
        )

        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-O0>
            $<$<CONFIG:Debug>:-g>
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:Release>:-DNDEBUG>
        )

        if(PT_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>
                $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
            )
            target_link_options(${target} PRIVATE
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>
            )
        endif()
    endif()
endfunction()

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Apply orthodox-C++23 compile/link flags to a single target.
# Per principles doc §08: no exceptions, no RTTI, hard warnings.
# ASan is opt-in via PT_ENABLE_SANITIZERS (mimalloc + ASan needs MI_TRACK_ASAN).
#
# Cross-platform: AppleClang/Clang/GCC use the canonical -flags;
# MSVC (and clang-cl in MSVC-driver mode) gets equivalent /flags.

# --- Sanitizer flags (project-wide) ----------------------------------------
# Enabled via the `linux-asan-ubsan-debug` preset (or any other config that
# defines PT_ENABLE_SANITIZERS=ON).  Registered with `add_compile_options` /
# `add_link_options` rather than per-target so the flags propagate to
# everything in the build graph -- including the doctest unit-test
# executables in tests/, which don't call `pt_set_compiler_options`
# (they're test binaries, not library targets, and don't want the strict
# -fno-exceptions / -fno-rtti / etc. policy applied at lib level).
#
# UBSan sub-checks omitted:
#   * `vptr` -- requires RTTI to identify polymorphic objects.  The
#     project compiles every TU with `-fno-rtti` (orthodox-C++23 policy
#     in pt_set_compiler_options below).  Enabling `vptr` would produce
#     a link-time complaint about a missing __ubsan_handle_dynamic_type
#     symbol and / or false positives on every devirtualised call.
#     Both Clang and GCC understand `-fno-sanitize=vptr`.
#   * `function` -- Clang-only sub-check (same RTTI requirement).  GCC
#     doesn't ship a `function` UBSan handler, and `-fno-sanitize=function`
#     under GCC produces "unrecognized argument" diagnostics that combined
#     with -Werror can fail the build.  Gate the opt-out on Clang.
#
# `-fno-sanitize-recover=all` makes every ASan / UBSan diagnostic fatal
# (default UBSan behaviour is "warn and keep going", which lets bugs
# pile up in a CI log without ever failing the job).  Combined with the
# CI job's `halt_on_error=1` env var it gives a single, fail-fast bug
# report on the first failure.
#
# `-O1 -g` is the canonical sanitizer optimisation level.  -O0 keeps
# the build readable but is so slow that a full ctest run takes ~10x
# longer than necessary; -O2/-O3 makes ASan's shadow-memory layout
# work but loses the inline-frame addresses that make UBSan reports
# debuggable.  -O1 + -fno-omit-frame-pointer hits both goals.
if(PT_ENABLE_SANITIZERS AND NOT MSVC)
    set(_pt_san_flags
        -fsanitize=address,undefined
        -fno-sanitize=vptr
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer
        -O1
        -g
    )
    set(_pt_san_link_flags
        -fsanitize=address,undefined
        -fno-sanitize=vptr
    )
    # `function` is Clang-only; adding it under GCC produces an
    # unrecognized-argument diagnostic.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND _pt_san_flags      -fno-sanitize=function)
        list(APPEND _pt_san_link_flags -fno-sanitize=function)
    endif()
    add_compile_options(${_pt_san_flags})
    add_link_options(${_pt_san_link_flags})
endif()

function(pt_set_compiler_options target)
    target_compile_features(${target} PUBLIC cxx_std_23)

    if(MSVC)
        # /EHs-c-  : disable C++ exception handling (assume noexcept everywhere)
        # /GR-     : disable RTTI
        # /W4      : warning level 4 (close to -Wall -Wextra)
        # /permissive- : tighten conformance (closer to -Wpedantic)
        # /wd4100  : unused parameter (mirrors -Wno-unused-parameter)
        # /Zc:__cplusplus : report the real __cplusplus value (MSVC defaults to 199711L)
        # /Zc:preprocessor : standards-conformant preprocessor. Required
        #     for __VA_OPT__ (C++20) which the LOG_INFO/PT_CVAR macros
        #     rely on -- the legacy MSVC preprocessor doesn't recognise
        #     __VA_OPT__ at all and chokes with C2146 / C2059.
        target_compile_options(${target} PRIVATE
            /EHs-c-
            /GR-
            /W4
            /permissive-
            /Zc:__cplusplus
            /Zc:preprocessor
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
            # /Z7 (debug info embedded in .obj) instead of /Zi
            # (separate .pdb written via mspdbsrv.exe).  /Zi forces
            # every parallel cl.exe to funnel writes through one
            # serialised PDB-server process, which on a 16-core
            # build effectively bottlenecks at ~2-3 cores worth of
            # throughput regardless of -j or jobs:0 settings.
            # /Z7 lets each TU compile fully independently --
            # typical 3-5x speedup on multicore Windows builds with
            # zero runtime impact and same debugger info quality.
            # Trade-off: .obj files are ~30% bigger, final linker
            # PDB roughly the same size.
            $<$<CONFIG:Debug>:/Z7>
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

        # Skip -O0 in sanitizer builds: the project-wide block above
        # already pushed -O1 -g (the canonical sanitizer opt level).
        # -O0 here would clobber that and slow ctest by ~10x with no
        # extra signal -- ASan shadow memory + UBSan stacktraces work
        # at -O1 just fine.
        if(NOT PT_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Debug>:-O0>
                $<$<CONFIG:Debug>:-g>
            )
        endif()
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:Release>:-DNDEBUG>
        )
    endif()
endfunction()

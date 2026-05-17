# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Third-party dependencies via FetchContent.
# Pin versions; use SYSTEM to suppress warnings from foreign headers.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# --- glm: vector math (header-only) ----------------------------------------
FetchContent_Declare(glm
    URL           https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
    URL_HASH      SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
    SYSTEM
)

# --- fmt: formatted output (until libc++ std::print is everywhere) ---------
FetchContent_Declare(fmt
    URL           https://github.com/fmtlib/fmt/archive/refs/tags/11.2.0.tar.gz
    URL_HASH      SHA256=bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af
    SYSTEM
)

# --- mimalloc: persistent heap allocator -----------------------------------
# We want the static lib only; do NOT override system malloc -- we call
# mi_malloc explicitly from PersistentHeap so we keep accounting honest.
set(MI_BUILD_SHARED  OFF CACHE BOOL "" FORCE)
set(MI_BUILD_STATIC  ON  CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT  OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(MI_OVERRIDE      OFF CACHE BOOL "" FORCE)
set(MI_INSTALL_TOPLEVEL OFF CACHE BOOL "" FORCE)
# Under PT_ENABLE_SANITIZERS, route mimalloc's internal heap accounting
# through ASan's __asan_poison_memory_region / __asan_unpoison_memory_region
# so ASan sees the same red zones mimalloc maintains internally. Without
# this, ASan flags every mi_free as a "double-free" because mimalloc's
# segment-cache reuses freed slots before ASan has marked them poisoned,
# and every mi_malloc as "use-after-free" symmetrically. MI_TRACK_ASAN is
# mimalloc 2.1+'s blessed integration for sanitizer builds.
if(PT_ENABLE_SANITIZERS)
    set(MI_TRACK_ASAN ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(mimalloc
    URL           https://github.com/microsoft/mimalloc/archive/refs/tags/v2.1.7.tar.gz
    URL_HASH      SHA256=0eed39319f139afde8515010ff59baf24de9e47ea316a315398e8027d198202d
    SYSTEM
)

# --- Tracy: profiler client (optional) -------------------------------------
if(PT_ENABLE_TRACY)
    set(TRACY_ENABLE     ON  CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND  ON  CACHE BOOL "" FORCE)  # only profile when client connects
    set(TRACY_NO_BROADCAST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(tracy
        URL           https://github.com/wolfpld/tracy/archive/refs/tags/v0.11.1.tar.gz
        URL_HASH      SHA256=2c11ca816f2b756be2730f86b0092920419f3dabc7a7173829ffd897d91888a1
        SYSTEM
    )
endif()

# --- glfw: window + input --------------------------------------------------
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    URL           https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH      SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    SYSTEM
)

# --- enkiTS: job system ----------------------------------------------------
# v1.11's TaskScheduler.cpp uses std::is_pod which C++23 removed.  Patch
# during fetch.  Also: enkiTS uses old-style include_directories(), so the
# include path doesn't propagate via link; we re-add it after MakeAvailable.
set(ENKITS_BUILD_C_INTERFACE OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_SHARED      OFF CACHE BOOL "" FORCE)
set(ENKITS_INSTALL           OFF CACHE BOOL "" FORCE)
FetchContent_Declare(enkits
    URL           https://github.com/dougbinks/enkiTS/archive/refs/tags/v1.11.tar.gz
    URL_HASH      SHA256=b57a782a6a68146169d29d180d3553bfecb9f1a0e87a5159082331920e7d297e
    SYSTEM
    PATCH_COMMAND  ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/patch_enkits.cmake
)

# --- civetweb: HTTP + WebSocket server -------------------------------------
set(CIVETWEB_ENABLE_WEBSOCKETS ON  CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SSL        OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_CXX        OFF CACHE BOOL "" FORCE)
set(CIVETWEB_BUILD_TESTING     OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_LUA        OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_DUKTAPE    OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_SERVER_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(CIVETWEB_INSTALL_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(CIVETWEB_ENABLE_ASAN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(civetweb
    URL           https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.tar.gz
    URL_HASH      SHA256=f0e471c1bf4e7804a6cfb41ea9d13e7d623b2bcc7bc1e2a4dd54951a24d60285
    SYSTEM
)

# --- tomlplusplus: TOML parsing (config files later phases) ----------------
FetchContent_Declare(tomlplusplus
    URL           https://github.com/marzer/tomlplusplus/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH      SHA256=8517f65938a4faae9ccf8ebb36631a38c1cadfb5efa85d9a72e15b9e97d25155
    SYSTEM
)

# --- VulkanMemoryAllocator: simplifies Vk allocation lifetime --------------
# Header-only.  Used by the Vulkan backend (P4+).
FetchContent_Declare(vma
    URL           https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.1.0.tar.gz
    URL_HASH      SHA256=ae134ecc37c55634f108e926f85d5d887b670360e77cd107affaf3a9539595f2
    SYSTEM
)

# --- nlohmann_json: WS protocol payloads -----------------------------------
# JSON_NOEXCEPTION turns throws into abort() so the library plays nice with
# our -fno-exceptions build.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
    URL           https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH      SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    SYSTEM
)

# --- embree: CPU BVH + ray-triangle intersection for the Software RHI -----
# Triangle-only BVH for the software backend's mesh path. We disable
# every geometry type except triangles, ray packets (single rays are
# fine here -- the engine fans out across pixels at the dispatcher),
# ISPC (no need for cross-platform SIMD codegen at this scale), and
# tutorials. Tasking system goes INTERNAL so we don't drag TBB into
# this codebase. ARM64 / Apple Silicon support has been first-class
# since Embree 4.x; the same source tree builds natively on
# x86_64 Windows + NVIDIA RTX too.
#
# Two acquisition paths, in order of preference:
#   1. Pre-built artefact from .github/workflows/prebuild-embree.yml,
#      consumed via cmake/EmbreeBinary.cmake.  Sets EMBREE_PREBUILT_FOUND
#      = TRUE and an `embree` IMPORTED STATIC target; skips the slow
#      from-source compile entirely (~30s download vs ~10-15min build).
#   2. Source compile via FetchContent.  Triggered when EmbreeBinary
#      couldn't find a prebuilt for the host platform / config -- new
#      Embree version, ISA flag bump, unsupported platform, etc.  The
#      first such build is slow; prebuild-embree.yml then runs and
#      uploads the artefact so the NEXT configure hits the fast path.
#
# Both paths read their settings from cmake/EmbreeConfig.cmake (single
# source of truth for version + flags) so they can't drift.
include(${CMAKE_CURRENT_LIST_DIR}/EmbreeBinary.cmake)
if(NOT EMBREE_PREBUILT_FOUND)
    pt_apply_embree_config()
    FetchContent_Declare(embree
        URL       ${EMBREE_VENDORED_URL}
        URL_HASH  ${EMBREE_VENDORED_URL_HASH}
        SYSTEM
    )
endif()

# --- manifold: mesh CSG (P9 headline) --------------------------------------
# Robust manifold-mesh boolean ops (union/intersect/subtract). Builds with
# CMake via FetchContent. We disable everything except the core C++ lib --
# no python bindings, no tests, no fuzzing, no native parallelism (TBB)
# since the job system will run booleans on a worker thread anyway.
set(MANIFOLD_TEST       OFF CACHE BOOL "" FORCE)
set(MANIFOLD_PYBIND     OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CBIND      OFF CACHE BOOL "" FORCE)
set(MANIFOLD_JSBIND     OFF CACHE BOOL "" FORCE)
set(MANIFOLD_DEBUG      OFF CACHE BOOL "" FORCE)
set(MANIFOLD_EXPORT     OFF CACHE BOOL "" FORCE)
set(MANIFOLD_DOWNLOADS  OFF CACHE BOOL "" FORCE)
set(MANIFOLD_PAR            OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CROSS_SECTION  OFF CACHE BOOL "" FORCE)
FetchContent_Declare(manifold
    URL           https://github.com/elalish/manifold/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH      SHA256=1e47f69a96fe228a953e6bfce99657b6d278ed98a822e950322f722adf2e74ed
    SYSTEM
)

# --- doctest: unit test framework (header-only) ----------------------------
# Single-header testing framework. Fast compile (the framework header
# itself is ~7000 lines but only the TU declaring DOCTEST_CONFIG_IMPLEMENT
# pays the framework-implementation parse cost), MIT licensed, supports
# the usual TEST_CASE / SUBCASE / CHECK / REQUIRE / SECTION idioms. Picked
# over Catch2 / GoogleTest specifically because the compile-time overhead
# is much lower -- meaningful when test TUs proliferate across the
# pt_math / pt_csg / pt_renderer / pt_console modules.
#
# Only fetched when PT_BUILD_TESTS is ON (default ON; CI release builds
# pass -DPT_BUILD_TESTS=OFF). Gating it here avoids paying the (small)
# FetchContent populate cost on configures that don't need tests.
if(PT_BUILD_TESTS)
    FetchContent_Declare(doctest
        URL           https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
        URL_HASH      SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
        SYSTEM
    )
endif()

FetchContent_MakeAvailable(glm fmt mimalloc glfw enkits civetweb tomlplusplus nlohmann_json)

if(PT_BUILD_TESTS)
    FetchContent_MakeAvailable(doctest)
endif()

# Cross-platform flags for silencing third-party warnings.  Vendored
# libraries (Embree's kernels, civetweb's sha1.inl, Tracy's sprintf use,
# etc.) emit thousands of warnings we can't fix without forking.  We
# silence them at target scope so our own code stays under strict
# warning levels.  Earlier versions of this file used GCC-style "-w" /
# "-UDEBUG" unconditionally, which MSVC silently ignored -- civetweb
# alone leaked ~137 C5045 warnings into the Windows release log on
# v0.3.15.  Use the matching MSVC flag form when CMAKE_CXX_COMPILER_ID
# is MSVC (cl.exe).  clang-cl reports as "Clang" so it falls through
# to the unix branch; clang-cl accepts "-w" / "-UDEBUG" in addition to
# the MSVC-style flags anyway, so either branch works for it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(PT_DEP_WARN_SILENCE_FLAG /w)
    # /U<symbol> with no space -- if we used `/U DEBUG` here, CMake's
    # set() would create a two-element list and target_compile_options
    # would forward `/U` and `DEBUG` as separate cl.exe args.  cl.exe
    # then treats `DEBUG` as a source-input filename rather than the
    # symbol to undefine, and the DEBUG macro stays defined.  Joined
    # form matches the documented MSVC syntax and the -UDEBUG semantics
    # on the GCC/Clang side below.
    set(PT_DEP_UNDEF_DEBUG_FLAGS /UDEBUG)
else()
    set(PT_DEP_WARN_SILENCE_FLAG -w)
    set(PT_DEP_UNDEF_DEBUG_FLAGS -UDEBUG)
endif()

# Embree compile + per-target tweaks below are only relevant when we're
# building Embree from source (EmbreeBinary.cmake didn't find a prebuilt
# for this host).  The prebuilt path is a single IMPORTED target with no
# in-tree sources to compile, so there's nothing to wrap in the C++17
# block and no sub-targets (sys / math / simd / lexers / tasking) to
# tweak warning flags on.
if(NOT EMBREE_PREBUILT_FOUND)
    # Embree's headers (common/sys/vector.h etc.) rely on pre-C++17 transitive
    # includes (<type_traits>, <exception>) that libc++ no longer pulls in
    # automatically when the consumer compiles with C++20+. Building Embree
    # itself in C++17 sidesteps the issue without forking; consumers of
    # Embree (our SoftwareDevice TU) keep building in the project's standard
    # C++23. Block scope (see Manifold pattern below) confines the override
    # so it doesn't leak to subsequent fetches.
    block()
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        FetchContent_MakeAvailable(embree)
    endblock()
    # Silence Embree's vendored warning-as-noise so the build log stays
    # readable; the lib has ~hundreds of warnings we can't fix without
    # forking the upstream project.
    if(TARGET embree)
        target_compile_options(embree PRIVATE ${PT_DEP_WARN_SILENCE_FLAG})
    endif()
    foreach(_t sys math simd lexers tasking)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE ${PT_DEP_WARN_SILENCE_FLAG})
        endif()
    endforeach()
endif()
if(PT_ENABLE_VULKAN_BACKEND)
    FetchContent_MakeAvailable(vma)
endif()

# Manifold reads the global TRACY_ENABLE cache var (same one we set for our
# Tracy fetch) and tries to download its own copy of tracy when it sees ON.
# Shadow it OFF for just manifold's configure step -- our renderer doesn't
# need manifold-internal profiling, and tracy still ships normally for our
# own targets via the unconditional fetch below.
block()
    set(TRACY_ENABLE OFF)
    FetchContent_MakeAvailable(manifold)
endblock()

# civetweb prints a flood of "*** ... worker_thread_run:..." trace lines
# whenever its `DEBUG` macro is defined (which CMake's Debug build type
# turns on). NDEBUG doesn't help -- the gate is `#if defined(DEBUG)`.
# Pass -UDEBUG to undefine it for this target only.
#
# civetweb uses C11 features (_Static_assert, etc.) but its CMakeLists
# doesn't request C11, so Apple Clang warns -Wpre-c11-compat on every
# build. Right fix is to give it the standard it actually uses, not
# silence the warning. C_STANDARD_REQUIRED ensures the compile fails
# loudly if a future bump pulls in an even newer C feature.
if(TARGET civetweb-c-library)
    # PT_DEP_UNDEF_DEBUG_FLAGS: kill civetweb's worker-thread trace flood
    #     under cmake Debug (DEBUG macro gate inside civetweb's source).
    # C_STANDARD 11: civetweb uses _Static_assert (a C11 feature).
    # PT_DEP_WARN_SILENCE_FLAG: civetweb's source has ~100 warnings (extra-
    #     semis in sha1.inl, sprintf deprecation, format mismatches, alloca
    #     usage on Mac; ~137 C5045 Spectre mitigation notes on MSVC) we
    #     can't fix without forking. Target-scoped suppresses only for
    #     this library; our own targets stay strict.
    target_compile_options(civetweb-c-library
        PRIVATE ${PT_DEP_UNDEF_DEBUG_FLAGS} ${PT_DEP_WARN_SILENCE_FLAG})
    set_target_properties(civetweb-c-library PROPERTIES
        C_STANDARD          11
        C_STANDARD_REQUIRED ON)
endif()
if(PT_ENABLE_TRACY)
    FetchContent_MakeAvailable(tracy)
    # Tracy uses deprecated sprintf in TracyProfiler/TracySocket/...
    # Same policy as civetweb -- vendored third-party C++ we don't
    # maintain, suppress at target scope.
    if(TARGET TracyClient)
        target_compile_options(TracyClient PRIVATE ${PT_DEP_WARN_SILENCE_FLAG})
    endif()
endif()

# enkiTS publishes its include path via the directory-scope command
# include_directories(), which is local to its own CMakeLists.txt.  Promote
# it to a target property so consumers picking up enkiTS via
# target_link_libraries actually see TaskScheduler.h.
if(TARGET enkiTS)
    target_include_directories(enkiTS PUBLIC ${enkits_SOURCE_DIR}/src)
endif()

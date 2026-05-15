# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Third-party dependencies via FetchContent.
# Pin versions; use SYSTEM to suppress warnings from foreign headers.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# --- glm: vector math (header-only) ----------------------------------------
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    ON
    SYSTEM
)

# --- fmt: formatted output (until libc++ std::print is everywhere) ---------
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.2.0
    GIT_SHALLOW    ON
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
FetchContent_Declare(mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG        v2.1.7
    GIT_SHALLOW    ON
    SYSTEM
)

# --- Tracy: profiler client (optional) -------------------------------------
if(PT_ENABLE_TRACY)
    set(TRACY_ENABLE     ON  CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND  ON  CACHE BOOL "" FORCE)  # only profile when client connects
    set(TRACY_NO_BROADCAST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        v0.11.1
        GIT_SHALLOW    ON
        SYSTEM
    )
endif()

# --- glfw: window + input --------------------------------------------------
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    ON
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
    GIT_REPOSITORY https://github.com/dougbinks/enkiTS.git
    GIT_TAG        v1.11
    GIT_SHALLOW    ON
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
    GIT_REPOSITORY https://github.com/civetweb/civetweb.git
    GIT_TAG        v1.16
    GIT_SHALLOW    ON
    SYSTEM
)

# --- tomlplusplus: TOML parsing (config files later phases) ----------------
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    ON
    SYSTEM
)

# --- VulkanMemoryAllocator: simplifies Vk allocation lifetime --------------
# Header-only.  Used by the Vulkan backend (P4+).
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    ON
    SYSTEM
)

# --- nlohmann_json: WS protocol payloads -----------------------------------
# JSON_NOEXCEPTION turns throws into abort() so the library plays nice with
# our -fno-exceptions build.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    ON
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
set(EMBREE_TUTORIALS                OFF CACHE BOOL   "" FORCE)
set(EMBREE_STATIC_LIB               ON  CACHE BOOL   "" FORCE)
set(EMBREE_ISPC_SUPPORT             OFF CACHE BOOL   "" FORCE)
set(EMBREE_TASKING_SYSTEM           "INTERNAL" CACHE STRING "" FORCE)
set(EMBREE_GEOMETRY_TRIANGLE        ON  CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_QUAD            OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_CURVE           OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_SUBDIVISION     OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_USER            OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_INSTANCE        ON  CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_INSTANCE_ARRAY  OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_GRID            OFF CACHE BOOL   "" FORCE)
set(EMBREE_GEOMETRY_POINT           OFF CACHE BOOL   "" FORCE)
set(EMBREE_RAY_PACKETS              OFF CACHE BOOL   "" FORCE)
set(EMBREE_FILTER_FUNCTION          OFF CACHE BOOL   "" FORCE)
set(EMBREE_BACKFACE_CULLING         OFF CACHE BOOL   "" FORCE)
set(EMBREE_ZIP_MODE                 OFF CACHE BOOL   "" FORCE)
FetchContent_Declare(embree
    GIT_REPOSITORY https://github.com/RenderKit/embree.git
    GIT_TAG        v4.4.0
    GIT_SHALLOW    ON
    SYSTEM
)

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
    GIT_REPOSITORY https://github.com/elalish/manifold.git
    GIT_TAG        v3.1.1
    GIT_SHALLOW    ON
    SYSTEM
)

FetchContent_MakeAvailable(glm fmt mimalloc glfw enkits civetweb tomlplusplus nlohmann_json)

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
    target_compile_options(embree PRIVATE -w)
endif()
foreach(_t sys math simd lexers tasking)
    if(TARGET ${_t})
        target_compile_options(${_t} PRIVATE -w)
    endif()
endforeach()
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
    # -UDEBUG: kill civetweb's worker-thread trace flood under cmake Debug.
    # C_STANDARD 11: civetweb uses _Static_assert (a C11 feature).
    # -w: civetweb's source has ~100 warnings (extra-semis in sha1.inl,
    #     sprintf deprecation, format mismatches, alloca usage) we
    #     can't fix without forking. Target-scoped -w suppresses only
    #     for this library; our own targets stay strict.
    target_compile_options(civetweb-c-library PRIVATE -UDEBUG -w)
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
        target_compile_options(TracyClient PRIVATE -w)
    endif()
endif()

# enkiTS publishes its include path via the directory-scope command
# include_directories(), which is local to its own CMakeLists.txt.  Promote
# it to a target property so consumers picking up enkiTS via
# target_link_libraries actually see TaskScheduler.h.
if(TARGET enkiTS)
    target_include_directories(enkiTS PUBLIC ${enkits_SOURCE_DIR}/src)
endif()

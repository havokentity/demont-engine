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
    PATCH_COMMAND  sed -i.bak "s/std::is_pod/std::is_trivially_copyable/g" src/TaskScheduler.cpp
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

FetchContent_MakeAvailable(glm fmt mimalloc glfw enkits civetweb tomlplusplus nlohmann_json)
if(PT_ENABLE_VULKAN_BACKEND)
    FetchContent_MakeAvailable(vma)
endif()

# civetweb prints a flood of "*** ... worker_thread_run:..." trace lines
# whenever its `DEBUG` macro is defined (which CMake's Debug build type
# turns on). NDEBUG doesn't help -- the gate is `#if defined(DEBUG)`.
# Pass -UDEBUG to undefine it for this target only.
if(TARGET civetweb-c-library)
    target_compile_options(civetweb-c-library PRIVATE -UDEBUG)
endif()
if(PT_ENABLE_TRACY)
    FetchContent_MakeAvailable(tracy)
endif()

# enkiTS publishes its include path via the directory-scope command
# include_directories(), which is local to its own CMakeLists.txt.  Promote
# it to a target property so consumers picking up enkiTS via
# target_link_libraries actually see TaskScheduler.h.
if(TARGET enkiTS)
    target_include_directories(enkiTS PUBLIC ${enkits_SOURCE_DIR}/src)
endif()

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

FetchContent_MakeAvailable(glm fmt mimalloc)
if(PT_ENABLE_TRACY)
    FetchContent_MakeAvailable(tracy)
endif()

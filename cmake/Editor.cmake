# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Build glue for the React+Vite editor under web/editor/. Two phases:
#
#   1. run `npm install && npm run build` (gated by PT_BUILD_EDITOR).
#      `npm install` is dependency-cached by package-lock.json; the
#      build re-runs whenever any source under web/editor/ (panels,
#      shared/, configs) changes. Output lands in web/editor/dist/.
#
#   2. walk dist/ and emit a single C++ file embedding every byte
#      under an `pt::editor::FindAsset(uri)` lookup. The engine's
#      ConsoleServer maps `GET /editor/<panel>` and `GET /panels/...
#      GET /shared/...` to this table.
#
# Both phases attach to the pt_engine static library so a clean
# `cmake --build` produces a self-contained demont binary.
#
# Opt-out: pass -DPT_BUILD_EDITOR=OFF (or have no node/npm installed).
# In that case the embedded asset table degrades to a stub HTML that
# says "run npm install && npm run build first".

option(PT_BUILD_EDITOR
    "Build the web/editor/ React shell and embed its dist/ tree into pt_engine"
    ON)

# Locate node + npm. If either is missing we set PT_EDITOR_ACTIVE=OFF
# and the embedded-stub path takes over -- the engine still links.
set(PT_EDITOR_ACTIVE OFF)
set(PT_EDITOR_DIR "${CMAKE_SOURCE_DIR}/web/editor")
set(PT_EDITOR_DIST_DIR "${PT_EDITOR_DIR}/dist")

if(PT_BUILD_EDITOR)
    find_program(NPM_EXECUTABLE
        NAMES npm npm.cmd
        HINTS
            /opt/homebrew/bin
            /usr/local/bin
            /usr/bin
            ENV PATH
        DOC "npm CLI used to build the React editor shell")
    if(NPM_EXECUTABLE)
        set(PT_EDITOR_ACTIVE ON)
        message(STATUS "Editor: npm at ${NPM_EXECUTABLE}")
    else()
        message(STATUS "Editor: npm not found -- editor will embed a build-me stub. "
                       "Install Node 20+ (https://nodejs.org/) and re-configure, "
                       "or pass -DPT_BUILD_EDITOR=OFF to silence this.")
    endif()
endif()

# Helper: add the editor build + embed steps to a target. The target
# (pt_console) gets:
#   - a dependency on a `pt_editor_build` custom_target that runs npm
#   - an additional source file (the generated assets.cpp) that
#     contains the embedded byte tables + FindAsset()
#
# The .cpp is regenerated whenever any file under web/editor/dist
# changes (the build's own outputs).
function(pt_attach_editor_build target)
    set(generated_dir "${CMAKE_BINARY_DIR}/editor_embedded")
    file(MAKE_DIRECTORY "${generated_dir}")
    set(editor_cpp "${generated_dir}/editor_assets.cpp")
    set(npm_stamp  "${generated_dir}/.npm_build.stamp")

    # Pull a list of editor source files so the npm build re-runs when
    # any of them change. Configure-time glob: any file added or
    # removed needs `cmake .` to re-run, which CMake auto-detects via
    # CONFIGURE_DEPENDS.
    file(GLOB_RECURSE _editor_srcs CONFIGURE_DEPENDS
        "${PT_EDITOR_DIR}/package.json"
        "${PT_EDITOR_DIR}/tsconfig.json"
        "${PT_EDITOR_DIR}/vite.config.ts"
        "${PT_EDITOR_DIR}/shared/src/*"
        "${PT_EDITOR_DIR}/shared/package.json"
        "${PT_EDITOR_DIR}/panels/*/index.html"
        "${PT_EDITOR_DIR}/panels/*/package.json"
        "${PT_EDITOR_DIR}/panels/*/src/*"
    )

    if(PT_EDITOR_ACTIVE)
        # npm install runs implicitly on the first build (if node_modules
        # is missing). Both steps happen in one command so the stamp file
        # only gets touched after the whole build succeeds.
        add_custom_command(
            OUTPUT "${npm_stamp}"
            COMMAND "${CMAKE_COMMAND}" -E echo "[editor] npm install + run build (sources changed)"
            COMMAND "${NPM_EXECUTABLE}" install --no-audit --no-fund --silent
            COMMAND "${NPM_EXECUTABLE}" run build --silent
            COMMAND "${CMAKE_COMMAND}" -E touch "${npm_stamp}"
            WORKING_DIRECTORY "${PT_EDITOR_DIR}"
            DEPENDS ${_editor_srcs}
            COMMENT "Building web/editor/ React shell"
            VERBATIM
        )
    else()
        # No npm available -- write the stamp so the next step still has
        # something to depend on. The embed step's INPUT_DIR won't exist,
        # so it'll emit the fallback stub.
        add_custom_command(
            OUTPUT "${npm_stamp}"
            COMMAND "${CMAKE_COMMAND}" -E touch "${npm_stamp}"
            COMMENT "Editor: skipping npm build (PT_EDITOR_ACTIVE=OFF)"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${editor_cpp}"
        COMMAND "${CMAKE_COMMAND}"
            -DINPUT_DIR=${PT_EDITOR_DIST_DIR}
            -DURL_PREFIX=${PT_EDITOR_DIST_DIR}
            -DOUTPUT=${editor_cpp}
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedEditorAssets.cmake"
        DEPENDS
            "${npm_stamp}"
            "${CMAKE_SOURCE_DIR}/cmake/EmbedEditorAssets.cmake"
        COMMENT "Embedding web/editor/dist/ into pt::editor::FindAsset"
        VERBATIM
    )

    target_sources(${target} PRIVATE "${editor_cpp}")
endfunction()

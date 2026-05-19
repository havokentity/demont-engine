# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Standalone CMake script (invoked via `cmake -P`) that walks an entire
# directory tree and emits ONE C++ source file containing:
#   - the bytes of every file as a static const unsigned char array
#   - a lookup table mapping URL path -> (data, size, mime)
#   - an exported function pt::editor::FindAsset(uri)
#
# Inputs (passed as -D flags):
#   INPUT_DIR   absolute path to scan (e.g. .../web/editor/dist)
#   URL_PREFIX  prefix to strip from each file's path to form its URL
#               (typically the same as INPUT_DIR, leaves only the
#               relative tail like /panels/scene-hierarchy/index.html)
#   OUTPUT      absolute path to the generated .cpp
#   SENTINEL    absolute path to a stamp file; touched on success
#
# The matching header lives at
# src/editor/EditorAssets.h (hand-written; declares the same
# FindAsset API). When INPUT_DIR doesn't exist (npm build was
# skipped), this still emits a valid .cpp file with an empty table
# and a single stub entry that returns a "build the editor first"
# HTML page so demont keeps linking even when web/editor/dist is
# missing.

if(NOT DEFINED INPUT_DIR)
    message(FATAL_ERROR "EmbedEditorAssets: INPUT_DIR not set")
endif()
if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "EmbedEditorAssets: OUTPUT not set")
endif()
if(NOT DEFINED URL_PREFIX)
    set(URL_PREFIX "${INPUT_DIR}")
endif()

# Helper -- map extension to MIME type. Kept tight to the set Vite
# actually emits + a couple of likely future additions (png/svg).
function(_mime_for_ext extn out_var)
    if(extn STREQUAL ".html")
        set(${out_var} "text/html; charset=utf-8" PARENT_SCOPE)
    elseif(extn STREQUAL ".js")
        set(${out_var} "application/javascript; charset=utf-8" PARENT_SCOPE)
    elseif(extn STREQUAL ".css")
        set(${out_var} "text/css; charset=utf-8" PARENT_SCOPE)
    elseif(extn STREQUAL ".map")
        # Source maps -- ship them so DevTools sees real symbol names.
        set(${out_var} "application/json; charset=utf-8" PARENT_SCOPE)
    elseif(extn STREQUAL ".json")
        set(${out_var} "application/json; charset=utf-8" PARENT_SCOPE)
    elseif(extn STREQUAL ".png")
        set(${out_var} "image/png" PARENT_SCOPE)
    elseif(extn STREQUAL ".svg")
        set(${out_var} "image/svg+xml" PARENT_SCOPE)
    elseif(extn STREQUAL ".ico")
        set(${out_var} "image/x-icon" PARENT_SCOPE)
    elseif(extn STREQUAL ".woff2")
        set(${out_var} "font/woff2" PARENT_SCOPE)
    else()
        set(${out_var} "application/octet-stream" PARENT_SCOPE)
    endif()
endfunction()

set(_have_files FALSE)
set(_table_entries "")
set(_blobs "")
set(_blob_idx 0)

if(EXISTS "${INPUT_DIR}" AND IS_DIRECTORY "${INPUT_DIR}")
    file(GLOB_RECURSE _files
        LIST_DIRECTORIES FALSE
        RELATIVE "${URL_PREFIX}"
        "${INPUT_DIR}/*")

    # Sort for determinism (file(GLOB) order isn't stable across runs).
    list(SORT _files)

    foreach(_rel IN LISTS _files)
        set(_abs "${URL_PREFIX}/${_rel}")
        if(NOT EXISTS "${_abs}")
            continue()
        endif()

        get_filename_component(_extn "${_rel}" EXT)
        _mime_for_ext("${_extn}" _mime)

        file(READ "${_abs}" _content HEX)
        string(LENGTH "${_content}" _hex_len)
        math(EXPR _byte_len "${_hex_len} / 2")

        # hex pairs -> "0xAB," sequence, one regex pass. Same trick as
        # cmake/EmbedFile.cmake (avoids O(N^2) string append).
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_content}")
        # Insert a newline every 16 entries so the generated cpp isn't a
        # 1 MB single line (slang/clangd choke on huge lines).
        set(_b "0x[0-9a-f][0-9a-f],")
        string(REGEX REPLACE "(${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b})" "\\1\n" _bytes "${_bytes}")
        unset(_b)

        set(_sym "blob_${_blob_idx}")
        math(EXPR _blob_idx "${_blob_idx} + 1")

        string(APPEND _blobs "static const unsigned char ${_sym}_data[] = {\n${_bytes}\n};\n")
        string(APPEND _blobs "static const unsigned long ${_sym}_size = ${_byte_len};\n\n")
        string(APPEND _table_entries
            "    { \"/${_rel}\", ${_sym}_data, ${_sym}_size, \"${_mime}\" },\n")

        set(_have_files TRUE)
    endforeach()
endif()

if(NOT _have_files)
    # Fallback stub HTML so the engine still links + warns the user.
    set(_stub "<!doctype html><html><head><title>DeMonT editor</title>"
              "<style>body{background:#06080d;color:#e7eaf2;font-family:sans-serif;"
              "padding:40px;line-height:1.5}h1{color:#00f0ff}code{background:#0c1018;"
              "padding:2px 6px;border-radius:3px;color:#ff3a8c}</style></head><body>"
              "<h1>Editor not built</h1><p>The React+Vite editor bundle isn't embedded "
              "in this build. From the repo root, run:</p>"
              "<pre><code>cd web/editor &amp;&amp; npm install &amp;&amp; npm run build</code></pre>"
              "<p>then reconfigure / rebuild demont.</p></body></html>")
    string(REPLACE ";" "" _stub "${_stub}")  # join the list back together
    # Hex-encode the stub
    string(LENGTH "${_stub}" _stub_len)
    set(_hex "")
    math(EXPR _stub_end "${_stub_len} - 1")
    set(_stub_bytes "")
    foreach(_i RANGE 0 ${_stub_end})
        string(SUBSTRING "${_stub}" ${_i} 1 _c)
        string(ASCII 0 _zero)
        # Convert to hex via printf-style format
        # We use string(ASCII) only to compare; easier path: write the
        # raw text directly into the .cpp as a char array.
    endforeach()
    # Simpler: write the stub as a C++ raw string. The generated source
    # ends up smaller too.
    set(_stub_cpp
"static const char kEditorMissingHtml[] = R\"PT_DEMONT_EDITOR_STUB(${_stub})PT_DEMONT_EDITOR_STUB\";
")
    set(_blobs "${_stub_cpp}")
    set(_table_entries
"    { \"/panels/scene-hierarchy/index.html\",
      reinterpret_cast<const unsigned char*>(kEditorMissingHtml),
      sizeof(kEditorMissingHtml) - 1,
      \"text/html; charset=utf-8\" },
    { \"/panels/inspector/index.html\",
      reinterpret_cast<const unsigned char*>(kEditorMissingHtml),
      sizeof(kEditorMissingHtml) - 1,
      \"text/html; charset=utf-8\" },
    { \"/panels/asset-browser/index.html\",
      reinterpret_cast<const unsigned char*>(kEditorMissingHtml),
      sizeof(kEditorMissingHtml) - 1,
      \"text/html; charset=utf-8\" },
    { \"/panels/toolbar/index.html\",
      reinterpret_cast<const unsigned char*>(kEditorMissingHtml),
      sizeof(kEditorMissingHtml) - 1,
      \"text/html; charset=utf-8\" },
")
endif()

# Generate the .cpp. The include uses the angle-bracket form so the
# project's `-I src` flag (target_include_directories(pt_editor PUBLIC
# ${CMAKE_SOURCE_DIR}/src)) resolves it from the source root, not
# from the generated file's location inside the build tree.
set(_header
"// Auto-generated by cmake/EmbedEditorAssets.cmake. Do not edit.
#include <editor/EditorAssets.h>
#include <cstddef>
#include <string_view>

namespace pt::editor {

namespace {

${_blobs}struct AssetEntry {
    const char* uri;
    const unsigned char* data;
    unsigned long size;
    const char* mime;
};

static constexpr AssetEntry kAssets[] = {
${_table_entries}};

constexpr std::size_t kAssetCount = sizeof(kAssets) / sizeof(kAssets[0]);

}  // namespace

const EmbeddedAsset* FindAsset(std::string_view uri) {
    for (std::size_t i = 0; i < kAssetCount; ++i) {
        const auto& e = kAssets[i];
        if (uri == e.uri) {
            // The on-disk static table holds a different POD type so
            // we can do the conversion lazily here without forcing a
            // global copy. Return a pointer into a thread-local view.
            thread_local EmbeddedAsset view;
            view.uri = e.uri;
            view.data = e.data;
            view.size = e.size;
            view.mime = e.mime;
            return &view;
        }
    }
    return nullptr;
}

std::size_t AssetCount() { return kAssetCount; }

EmbeddedAsset AssetAt(std::size_t index) {
    if (index >= kAssetCount) return {};
    const auto& e = kAssets[index];
    EmbeddedAsset out;
    out.uri = e.uri;
    out.data = e.data;
    out.size = e.size;
    out.mime = e.mime;
    return out;
}

}  // namespace pt::editor
")

file(WRITE "${OUTPUT}" "${_header}")

if(DEFINED SENTINEL)
    file(WRITE "${SENTINEL}" "ok\n")
endif()

if(_have_files)
    message(STATUS "EmbedEditorAssets: emitted ${_blob_idx} files into ${OUTPUT}")
else()
    message(STATUS "EmbedEditorAssets: web/editor/dist missing -- emitted fallback stub")
endif()

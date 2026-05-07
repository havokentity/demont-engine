# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
add_library(pt_console STATIC
    Console.cpp
    ConsoleServer.cpp
)

target_include_directories(pt_console PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(pt_console
    PUBLIC  pt_core
    PUBLIC  nlohmann_json::nlohmann_json
    PRIVATE civetweb-c-library
)

# nlohmann/json: replace exception throws with std::abort so we can build
# with -fno-exceptions cleanly.
target_compile_definitions(pt_console PUBLIC JSON_NOEXCEPTION=1)

# Embed the web UI (index.html, console.js, console.css) as binary blobs.
pt_embed_resource(pt_console SYMBOL web_index_html  FILE web/index.html)
pt_embed_resource(pt_console SYMBOL web_console_js  FILE web/console.js)
pt_embed_resource(pt_console SYMBOL web_console_css FILE web/console.css)

pt_set_compiler_options(pt_console)

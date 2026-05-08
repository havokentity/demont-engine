# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# pt_compile_slang_module(SOURCE <file.slang>)
#
#   Compiles a Slang source file to a target-agnostic IR module
#   (`<name>.slang-module`) in `${CMAKE_BINARY_DIR}/shaders`. Modules
#   are referenced by entry-point shaders via `import <name>;` and
#   resolved at compile time by passing the shader output dir as a
#   `-I` search path. Module compiles run in parallel with each
#   other (independent ninja jobs) and with non-dependent shader
#   compiles, so the helpers no longer get re-parsed every time the
#   entry-point shader rebuilds.
#
#   Idempotent: safe to call from multiple shader trees (vulkan +
#   metal). The custom-command output is keyed on the module path so
#   ninja deduplicates.
#
# pt_compile_slang(TARGET <tgt>
#                  SOURCE <file.slang>
#                  STAGE  <compute|vertex|fragment>   default: compute
#                  ENTRY  <entry_point>               default: main
#                  TARGETS metal [spirv ...]
#                  MODULE_DEPS <name1> [<name2> ...])
#
# For each requested target, emits a build-time custom command that runs
# slangc and embeds the result as a binary blob into <tgt>. Symbol names
# follow the pattern shader_<name>_<format>{_data,_size}.
#
# MODULE_DEPS lists Slang modules (without extension) that the entry
# point imports. Each `<name>` resolves to `<binary>/shaders/<name>.slang-module`
# and gets added to the slangc invocation's DEPENDS so ninja waits
# for the module compile before building the entry-point shader.
#
# We deliberately compile Slang -> MSL *source* (not .metallib) on macOS,
# so we don't need Apple's separate Metal Toolchain installed; Metal does
# the source -> library compile at runtime via newLibrary().

function(pt_compile_slang_module)
    cmake_parse_arguments(M "" "SOURCE" "" ${ARGN})
    if(NOT M_SOURCE)
        message(FATAL_ERROR "pt_compile_slang_module needs SOURCE")
    endif()
    get_filename_component(name "${M_SOURCE}" NAME_WE)
    get_filename_component(full "${M_SOURCE}" ABSOLUTE)
    set(out_dir "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${out_dir}")
    set(out "${out_dir}/${name}.slang-module")
    if(NOT TARGET slang_module_${name})
        add_custom_command(
            OUTPUT  "${out}"
            COMMAND ${PT_SLANGC_BIN}
                    "${full}"
                    -emit-ir
                    -Wno-40100
                    -o       "${out}"
            DEPENDS "${full}" "${PT_SLANGC_BIN}"
            VERBATIM
            COMMENT "slangc module ${name}.slang -> .slang-module"
        )
        add_custom_target(slang_module_${name} DEPENDS "${out}")
    endif()
endfunction()

function(pt_compile_slang)
    cmake_parse_arguments(SLG "" "TARGET;SOURCE;STAGE;ENTRY" "TARGETS;MODULE_DEPS" ${ARGN})

    if(NOT SLG_TARGET OR NOT SLG_SOURCE OR NOT SLG_TARGETS)
        message(FATAL_ERROR "pt_compile_slang needs TARGET / SOURCE / TARGETS")
    endif()
    if(NOT SLG_STAGE)
        set(SLG_STAGE compute)
    endif()
    if(NOT SLG_ENTRY)
        set(SLG_ENTRY main)
    endif()

    get_filename_component(slg_name "${SLG_SOURCE}" NAME_WE)
    get_filename_component(slg_full "${SLG_SOURCE}" ABSOLUTE)

    set(out_dir "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${out_dir}")

    # Module dependencies: each MODULE_DEPS entry is a name that maps
    # to <out_dir>/<name>.slang-module. These get added to DEPENDS so
    # ninja waits for the module to be ready before this entry-point
    # compile starts. The slangc -I flag points slangc's `import`
    # resolver at out_dir so it picks up the precompiled IR file.
    set(module_outputs "")
    foreach(m ${SLG_MODULE_DEPS})
        list(APPEND module_outputs "${out_dir}/${m}.slang-module")
    endforeach()

    foreach(t ${SLG_TARGETS})
        set(slang_defs "")
        if(t STREQUAL "metal")
            set(ext "metal")
            set(slang_target "metal")
            list(APPEND slang_defs "-DPT_TARGET_METAL")
        elseif(t STREQUAL "spirv")
            set(ext "spv")
            set(slang_target "spirv")
            # Path tracer's push-constant block is too large for native
            # Vulkan's 256B hw limit; the shader checks PT_TARGET_SPIRV
            # to spill the tail into a UBO at vk::binding(14, 0).
            list(APPEND slang_defs "-DPT_TARGET_SPIRV")
        elseif(t STREQUAL "cpp")
            set(ext "cpp")
            set(slang_target "cpp")
        else()
            message(FATAL_ERROR "pt_compile_slang: unknown target '${t}'")
        endif()

        set(out "${out_dir}/${slg_name}.${ext}")
        # -Wno-40100 silences slangc's harmless "entry point 'main' has
        # been renamed to 'main_0'" notice -- it's a side effect of
        # Slang's mangling rules and not something we can avoid in
        # the source.
        add_custom_command(
            OUTPUT  "${out}"
            COMMAND ${PT_SLANGC_BIN}
                    "${slg_full}"
                    -target  ${slang_target}
                    -entry   ${SLG_ENTRY}
                    -stage   ${SLG_STAGE}
                    ${slang_defs}
                    -I       "${out_dir}"
                    -Wno-40100
                    -o       "${out}"
            DEPENDS "${slg_full}" "${PT_SLANGC_BIN}" ${module_outputs}
            VERBATIM
            COMMENT "slangc ${slg_name}.slang -> ${ext}"
        )

        set(symbol "shader_${slg_name}_${t}")
        pt_embed_resource(${SLG_TARGET}
            SYMBOL    ${symbol}
            FULL_PATH "${out}"
        )
    endforeach()
endfunction()

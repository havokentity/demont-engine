# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# pt_compile_slang(TARGET <tgt>
#                  SOURCE <file.slang>
#                  STAGE  <compute|vertex|fragment>   default: compute
#                  ENTRY  <entry_point>               default: main
#                  TARGETS metal [spirv ...])
#
# For each requested target, emits a build-time custom command that runs
# slangc and embeds the result as a binary blob into <tgt>. Symbol names
# follow the pattern shader_<name>_<format>{_data,_size}.
#
# We deliberately compile Slang -> MSL *source* (not .metallib) on macOS,
# so we don't need Apple's separate Metal Toolchain installed; Metal does
# the source -> library compile at runtime via newLibrary().

function(pt_compile_slang)
    cmake_parse_arguments(SLG "" "TARGET;SOURCE;STAGE;ENTRY" "TARGETS" ${ARGN})

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
                    -Wno-40100
                    -o       "${out}"
            DEPENDS "${slg_full}" "${PT_SLANGC_BIN}"
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

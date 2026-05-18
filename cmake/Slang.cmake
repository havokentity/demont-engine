# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte

# --- SDF Phase 2 (#98) megakernel-perf gates -------------------------------
#
# Phase 2 procedural SDFs (sdf_displace_noise / twist / bend / repeat /
# repeat-limited) plus their Dual3 forward-mode autodiff path are guarded
# inside SdfPrimitives.slang by PT_SDF_PROCEDURAL_OPS and PT_SDF_AUTODIFF.
# Default is OFF for both: a perf bisect on the default scene found a
# ~6 ms/frame regression introduced by the Phase 2 block when its mere
# presence got inlined into PathTrace's megakernel sphere-trace dispatch.
#
# Builds that need procedural SDFs flip PT_SDF_PROCEDURAL_OPS to ON; if
# they also want the cheaper forward-AD normal path, flip
# PT_SDF_AUTODIFF on top of that. Flags are CACHE variables so they
# show up in cmake-gui / ccmake and survive across configures. Both
# rhi_metal and rhi_vulkan pick them up by way of pt_compile_slang
# auto-appending them to every Slang invocation -- the module compile
# and the entry-point compile share the same preprocessor state, which
# Slang's IR linker depends on.
option(PT_SDF_PROCEDURAL_OPS
    "Compile Phase 2 procedural SDF ops + walkers into SdfPrimitives.slang. \
Default OFF -- presence alone costs ~6 ms/frame on the default scene because \
Slang's megakernel inlines every branch."
    OFF)
option(PT_SDF_AUTODIFF
    "Compile Dual3 forward-mode autodiff normal path into SdfPrimitives.slang. \
Default OFF. Requires PT_SDF_PROCEDURAL_OPS=ON to do anything; otherwise the \
gate inside SdfPrimitives.slang demotes this to OFF automatically."
    OFF)

# Build the slangc -D... fragment from the cache options. Both pt_compile_slang
# and pt_compile_slang_module append this to every invocation so module +
# entry-point preprocessor state stays consistent (Slang's IR linker requires
# this). Phase 1 analytic CSG is always available; these only toggle the
# Phase 2 procedural + Dual3 blocks inside SdfPrimitives.slang.
function(_pt_sdf_defines outVar)
    set(_d "")
    if(PT_SDF_PROCEDURAL_OPS)
        list(APPEND _d "-DPT_SDF_PROCEDURAL_OPS=1")
    endif()
    if(PT_SDF_AUTODIFF)
        list(APPEND _d "-DPT_SDF_AUTODIFF=1")
    endif()
    set(${outVar} "${_d}" PARENT_SCOPE)
endfunction()

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
#                  MODULE_DEPS <name1> [<name2> ...]
#                  EXTRA_DEFINES <-DFOO> [-DBAR ...]
#                  VARIANT  <suffix>)
#
# For each requested target, emits a build-time custom command that runs
# slangc and embeds the result as a binary blob into <tgt>. Symbol names
# follow the pattern shader_<name>_<format>{_data,_size}, OR --
# when VARIANT is supplied -- shader_<name>_<variant>_<format>{_data,_size},
# letting one .slang file produce multiple SPIR-V/MSL outputs with
# different preprocessor states (e.g. RT-on vs RT-off PathTrace builds
# for backends that lack VK_KHR_ray_query).
#
# EXTRA_DEFINES lets the caller append -D flags on top of the
# target-specific defaults (-DPT_TARGET_METAL / -DPT_TARGET_SPIRV).
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
    cmake_parse_arguments(M "" "SOURCE" "EXTRA_DEFINES" ${ARGN})
    if(NOT M_SOURCE)
        message(FATAL_ERROR "pt_compile_slang_module needs SOURCE")
    endif()
    get_filename_component(name "${M_SOURCE}" NAME_WE)
    get_filename_component(full "${M_SOURCE}" ABSOLUTE)
    set(out_dir "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${out_dir}")
    set(out "${out_dir}/${name}.slang-module")
    # EXTRA_DEFINES bakes module-level preprocessor state into the IR
    # blob. Entry-point shaders that import this module MUST be compiled
    # with the same defines (their EXTRA_DEFINES list should mirror the
    # module's), otherwise Slang's IR linker sees disagreeing symbol
    # tables. The SDF Phase 2 (#98) PT_SDF_PROCEDURAL_OPS / PT_SDF_AUTODIFF
    # cache options are auto-appended so both rhi_metal + rhi_vulkan
    # module compiles stay in sync without per-backend CMakeLists edits.
    _pt_sdf_defines(_sdf_defs)
    if(NOT TARGET slang_module_${name})
        add_custom_command(
            OUTPUT  "${out}"
            COMMAND ${PT_SLANGC_BIN}
                    "${full}"
                    -emit-ir
                    ${_sdf_defs}
                    ${M_EXTRA_DEFINES}
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
    cmake_parse_arguments(SLG "" "TARGET;SOURCE;STAGE;ENTRY;VARIANT" "TARGETS;MODULE_DEPS;EXTRA_DEFINES" ${ARGN})

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

        # When VARIANT is supplied the output filename and embedded
        # symbol both gain the suffix so a single .slang source can
        # produce multiple SPIR-V/MSL blobs that coexist in the same
        # binary (see PathTrace's rq / norq pair).
        if(SLG_VARIANT)
            set(out "${out_dir}/${slg_name}_${SLG_VARIANT}.${ext}")
            set(symbol "shader_${slg_name}_${SLG_VARIANT}_${t}")
            set(label "${slg_name}_${SLG_VARIANT}.slang")
        else()
            set(out "${out_dir}/${slg_name}.${ext}")
            set(symbol "shader_${slg_name}_${t}")
            set(label "${slg_name}.slang")
        endif()
        # -Wno-40100 silences slangc's harmless "entry point 'main' has
        # been renamed to 'main_0'" notice -- it's a side effect of
        # Slang's mangling rules and not something we can avoid in
        # the source.
        # The SDF Phase 2 cache-option defines are auto-appended so
        # entry-point compiles stay consistent with the SdfPrimitives
        # module compile (Slang's IR linker requires matching macro
        # state across module + entry point).
        _pt_sdf_defines(_sdf_defs)
        add_custom_command(
            OUTPUT  "${out}"
            COMMAND ${PT_SLANGC_BIN}
                    "${slg_full}"
                    -target  ${slang_target}
                    -entry   ${SLG_ENTRY}
                    -stage   ${SLG_STAGE}
                    ${slang_defs}
                    ${_sdf_defs}
                    ${SLG_EXTRA_DEFINES}
                    -I       "${out_dir}"
                    -Wno-40100
                    -o       "${out}"
            DEPENDS "${slg_full}" "${PT_SLANGC_BIN}" ${module_outputs}
            VERBATIM
            COMMENT "slangc ${label} -> ${ext}"
        )

        pt_embed_resource(${SLG_TARGET}
            SYMBOL    ${symbol}
            FULL_PATH "${out}"
        )
    endforeach()
endfunction()

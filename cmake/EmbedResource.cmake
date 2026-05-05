# pt_embed_resource(target SYMBOL symbol_name FILE relative_path)
# Generates an auto-built .cpp containing the file's bytes and adds it to
# the target. Two extern symbols become available at link time:
#   extern "C" const unsigned char <symbol>_data[];
#   extern "C" const unsigned long <symbol>_size;

function(pt_embed_resource target)
    cmake_parse_arguments(EMB "" "SYMBOL;FILE" "" ${ARGN})

    if(NOT EMB_SYMBOL OR NOT EMB_FILE)
        message(FATAL_ERROR "pt_embed_resource needs SYMBOL and FILE")
    endif()

    set(input_full ${CMAKE_SOURCE_DIR}/${EMB_FILE})
    set(generated_dir ${CMAKE_BINARY_DIR}/embedded)
    set(output_path ${generated_dir}/${EMB_SYMBOL}.cpp)

    file(MAKE_DIRECTORY ${generated_dir})

    add_custom_command(
        OUTPUT  ${output_path}
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${input_full}
                -DOUTPUT=${output_path}
                -DSYMBOL=${EMB_SYMBOL}
                -P ${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake
        DEPENDS ${input_full} ${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake
        VERBATIM
    )

    target_sources(${target} PRIVATE ${output_path})
endfunction()

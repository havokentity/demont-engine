# Helper invoked by add_custom_command. Reads ${INPUT}, writes a C++ source
# at ${OUTPUT} exposing two extern symbols:
#   const unsigned char ${SYMBOL}_data[];
#   const unsigned long ${SYMBOL}_size;

file(READ "${INPUT}" content HEX)
string(LENGTH "${content}" hex_len)
math(EXPR byte_len "${hex_len} / 2")

string(REGEX MATCHALL ".." pairs "${content}")
set(bytes "")
set(col 0)
foreach(p ${pairs})
    string(APPEND bytes "0x${p},")
    math(EXPR col "${col} + 1")
    if(col EQUAL 16)
        string(APPEND bytes "\n")
        set(col 0)
    endif()
endforeach()

file(WRITE "${OUTPUT}" "// Auto-generated. Do not edit.\n")
file(APPEND "${OUTPUT}" "extern \"C\" {\n")
# `extern` on the definition prevents the C++ default of giving const
# namespace-scope variables internal linkage -- without it the compiler
# can (and does, with -fvisibility=hidden) elide them entirely.
file(APPEND "${OUTPUT}" "extern const unsigned char ${SYMBOL}_data[];\n")
file(APPEND "${OUTPUT}" "extern const unsigned long ${SYMBOL}_size;\n")
file(APPEND "${OUTPUT}" "extern const unsigned char ${SYMBOL}_data[] = {\n${bytes}};\n")
file(APPEND "${OUTPUT}" "extern const unsigned long ${SYMBOL}_size = ${byte_len};\n")
file(APPEND "${OUTPUT}" "}\n")

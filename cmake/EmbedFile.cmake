# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Helper invoked by add_custom_command. Reads ${INPUT}, writes a C++ source
# at ${OUTPUT} exposing two extern symbols:
#   const unsigned char ${SYMBOL}_data[];
#   const unsigned long ${SYMBOL}_size;

file(READ "${INPUT}" content HEX)
string(LENGTH "${content}" hex_len)
math(EXPR byte_len "${hex_len} / 2")

# Convert hex pairs -> "0xAB,0xCD,..." in a single regex pass.  The
# previous implementation looped foreach over the pairs list and
# string(APPEND)ed each one onto a growing buffer -- that's O(N^2)
# in CMake script (every APPEND reallocates) and on a 490 KB file
# (245k pairs) it took ~3 minutes.  One regex pass is O(N).
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${content}")

# Insert a newline every 16 entries so the emitted cpp isn't one
# multi-megabyte line.  Some compilers / IDEs (notably older
# clangd / VS source viewers) slow to a crawl or hit line-length
# limits when parsing single-line files in the multi-MB range.
# Second regex pass; still O(N).  The {16} quantifier doesn't fire
# reliably in cmake's regex engine, so spell out the 16 hex pairs
# explicitly -- ugly but reliable.
set(_b "0x[0-9a-f][0-9a-f],")
string(REGEX REPLACE "(${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b}${_b})" "\\1\n" bytes "${bytes}")
unset(_b)

# `extern` on the definition prevents the C++ default of giving const
# namespace-scope variables internal linkage -- without it the
# compiler can (and does, with -fvisibility=hidden) elide them
# entirely.
set(prelude
"// Auto-generated. Do not edit.
extern \"C\" {
extern const unsigned char ${SYMBOL}_data[];
extern const unsigned long ${SYMBOL}_size;
extern const unsigned char ${SYMBOL}_data[] = {
")
set(epilogue
"};
extern const unsigned long ${SYMBOL}_size = ${byte_len};
}
")
file(WRITE "${OUTPUT}" "${prelude}${bytes}\n${epilogue}")

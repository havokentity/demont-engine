# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# `cmake -P` script invoked by missing-golden ctest cells in
# tests/CMakeLists.txt (golden-image regression matrix, issue #45).
#
# Why a separate script: the original implementation used
# `${CMAKE_COMMAND} -E echo "..."`, but `cmake -E echo` always exits 0.
# Combined with `WILL_FAIL TRUE` on the test (which inverts the
# pass/fail logic so a non-zero exit means PASS) the cell was treated
# as FAILED by ctest, exactly the opposite of the intent. ctest needs
# this to exit non-zero so WILL_FAIL flips it to PASS, while still
# printing a clear regen hint to the maintainer.
#
# Inputs (passed via -D ... on the cmake -P command line):
#   GOLDEN  -- absolute path to the missing golden PNG
#   CELL    -- ctest cell name (golden_<scene>__<backend>__<denoiser>)
#   ACTUAL  -- absolute path to the actual PNG that ctest's _render
#              step is producing on this host.

message("MISSING GOLDEN: ${GOLDEN}")
message("Regenerate with:")
message("  ctest -R ^${CELL}_render$")
message("  cp ${ACTUAL} ${GOLDEN}")
message("See Raytracer Plan/TESTING_STRATEGY.md for the full regen workflow.")
message(FATAL_ERROR "missing-golden placeholder for cell ${CELL}")

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
# Portable replacement for `sed -i.bak s/std::is_pod/std::is_trivially_copyable/g`
# Run via `cmake -P` so it works on Windows (no sed), Mac, and Linux alike.
# WORKING_DIRECTORY is the enkiTS source dir (set by FetchContent PATCH_COMMAND).

set(_file "src/TaskScheduler.cpp")
file(READ "${_file}" _content)
string(REPLACE "std::is_pod" "std::is_trivially_copyable" _content "${_content}")
file(WRITE "${_file}" "${_content}")

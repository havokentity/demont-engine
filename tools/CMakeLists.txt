# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Standalone developer tooling. Each tool is its own subdirectory + own
# executable target, intentionally not linked to the engine libraries
# so they build fast and run anywhere (no Vulkan SDK / Metal / etc.
# needed). Gated as a whole by PT_BUILD_TOOLS at the root level.

add_subdirectory(imgdiff)
add_subdirectory(pt_render_one_frame)

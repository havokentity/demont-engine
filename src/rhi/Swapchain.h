#pragma once

#include "Handles.h"

namespace pt::rhi {

// Swapchain abstraction. Currently all backends own their swapchain via
// the Device, so this header just defines the per-frame contract.
struct FrameContext {
    TextureHandle swapchain_image;   // write the final image here
    std::uint32_t width  = 0;
    std::uint32_t height = 0;
    std::uint32_t frame_index = 0;   // monotonically increasing
};

}  // namespace pt::rhi

#include "SoftwareDevice.h"

#include "../core/Log.h"
#include "../core/Memory/MemTag.h"
#include "../core/Memory/Memory.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>

#include <cstring>

namespace pt::rhi::sw {

// ---------- SoftwarePipeline ------------------------------------------------

SoftwarePipeline::SoftwarePipeline(std::string name) : name_(std::move(name)) {}

void SoftwarePipeline::SetClearColor(float r, float g, float b, float a) {
    clear_[0] = r; clear_[1] = g; clear_[2] = b; clear_[3] = a;
}
void SoftwarePipeline::GetClearColor(float& r, float& g, float& b, float& a) const {
    r = clear_[0]; g = clear_[1]; b = clear_[2]; a = clear_[3];
}

// ---------- SoftwareCommandBuffer -----------------------------------------

void SoftwareCommandBuffer::BindComputePipeline(PipelineHandle p) {
    bound_pipeline = p;
}

void SoftwareCommandBuffer::PushConstants(const void* data, std::size_t size) {
    if (size > sizeof(push_constants_buf)) size = sizeof(push_constants_buf);
    std::memcpy(push_constants_buf, data, size);
    push_constants_size = size;
}

void SoftwareCommandBuffer::Dispatch(std::uint32_t /*gx*/, std::uint32_t /*gy*/,
                                     std::uint32_t /*gz*/) {
    auto* pl = device_->GetPipeline(bound_pipeline);
    if (pl == nullptr) return;
    if (pl->Name() == "clear" && push_constants_size >= sizeof(float) * 4) {
        const auto* c = reinterpret_cast<const float*>(push_constants_buf);
        device_->StashClear(c[0], c[1], c[2], c[3]);
    }
}

// ---------- SoftwareDevice -------------------------------------------------

SoftwareDevice::SoftwareDevice(GLFWwindow* window) : window_(window) {
    pt::mem::TagScope scope(pt::MemTag::GpuBuffers);
    cmd_buf_ = std::make_unique<SoftwareCommandBuffer>(this);

    if (window_ != nullptr) {
        glfwGetFramebufferSize(window_, &width_, &height_);
        // The window must have been created with an OpenGL context for
        // this backend.  The Engine handles that on backend switch.
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);
        LOG_INFO("Software device online: {}x{} via GL context", width_, height_);
    }
}

SoftwareDevice::~SoftwareDevice() {
    if (window_ != nullptr) {
        glfwMakeContextCurrent(nullptr);
    }
}

BufferHandle SoftwareDevice::CreateBuffer(const BufferDesc& d) {
    bytes_held_.fetch_add(d.size, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    return BufferHandle{ next_id_++ };
}

TextureHandle SoftwareDevice::CreateTexture(const TextureDesc& d) {
    auto bytes = static_cast<std::size_t>(d.width) * d.height * 4;  // assume RGBA8
    bytes_held_.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard lock(resource_mutex_);
    return TextureHandle{ next_id_++ };
}

PipelineHandle SoftwareDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    pt::mem::TagScope scope(pt::MemTag::Render);
    auto pl = std::make_unique<SoftwarePipeline>(std::string(d.kernel_name));
    std::lock_guard lock(resource_mutex_);
    auto id = next_id_++;
    pipelines_.emplace(id, std::move(pl));
    return PipelineHandle{ id };
}

void SoftwareDevice::DestroyBuffer(BufferHandle)     { /* trivial in P2 */ }
void SoftwareDevice::DestroyTexture(TextureHandle)   { /* trivial in P2 */ }

void SoftwareDevice::DestroyPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    pipelines_.erase(h.id);
}

FrameContext SoftwareDevice::BeginFrame() {
    if (window_ != nullptr) {
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        if (w != width_ || h != height_) {
            width_ = w; height_ = h;
        }
    }
    return FrameContext{
        .swapchain_image = TextureHandle{0},
        .width  = static_cast<std::uint32_t>(width_),
        .height = static_cast<std::uint32_t>(height_),
        .frame_index = frame_index_,
    };
}

void SoftwareDevice::EndFrame(CommandBuffer*) {
    if (window_ == nullptr) return;
    glViewport(0, 0, width_, height_);
    glClearColor(pending_clear_[0], pending_clear_[1],
                 pending_clear_[2], pending_clear_[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window_);
    ++frame_index_;
}

CommandBuffer* SoftwareDevice::AcquireCommandBuffer() { return cmd_buf_.get(); }

void SoftwareDevice::Submit(CommandBuffer*) { /* command played back at Dispatch */ }

void SoftwareDevice::Resize(int w, int h) { width_ = w; height_ = h; }

std::size_t SoftwareDevice::CurrentAllocatedBytes() const {
    return bytes_held_.load(std::memory_order_relaxed);
}

void SoftwareDevice::StashClear(float r, float g, float b, float a) {
    pending_clear_[0] = r; pending_clear_[1] = g;
    pending_clear_[2] = b; pending_clear_[3] = a;
}

SoftwarePipeline* SoftwareDevice::GetPipeline(PipelineHandle h) {
    std::lock_guard lock(resource_mutex_);
    auto it = pipelines_.find(h.id);
    return it == pipelines_.end() ? nullptr : it->second.get();
}

}  // namespace pt::rhi::sw

namespace pt::rhi {

std::unique_ptr<Device> CreateSoftwareDevice(const NativeWindowHandle& w) {
    auto* gw = static_cast<GLFWwindow*>(w.opaque);
    return std::make_unique<sw::SoftwareDevice>(gw);
}

}  // namespace pt::rhi

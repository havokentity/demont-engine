#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace pt::app {

enum class GraphicsApi {
    None,    // backend will attach its own surface (Metal / Vulkan)
    OpenGL,  // GLFW manages an OpenGL Core 3.3 context (software backend)
};

class Window {
public:
    Window();
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool Create(int w, int h, std::string_view title,
                GraphicsApi api = GraphicsApi::None);
    void Destroy();

    // Tear down and recreate with a new graphics-API hint, preserving the
    // current size + (best effort) position.  Used by the Engine on
    // backend switch.
    bool Recreate(GraphicsApi api);

    void PollEvents();
    bool ShouldClose() const;
    void RequestClose();

    int  Width()  const noexcept { return width_;  }
    int  Height() const noexcept { return height_; }

    GLFWwindow* Handle() const noexcept { return handle_; }

    // Returns the platform-native handle (NSWindow* on macOS, HWND on
    // Windows).  Currently unused outside of future RHI backends.
    void* NativeHandle() const;

private:
    static void OnResize(GLFWwindow* w, int width, int height);
    static void OnKey(GLFWwindow* w, int key, int scancode, int action, int mods);

    GLFWwindow* handle_ = nullptr;
    int width_  = 0;
    int height_ = 0;
    GraphicsApi api_ = GraphicsApi::None;
    std::string title_;
};

}  // namespace pt::app

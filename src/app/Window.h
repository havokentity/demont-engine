#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace pt::app {

class Window {
public:
    Window();
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool Create(int w, int h, std::string_view title);
    void Destroy();

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
};

}  // namespace pt::app

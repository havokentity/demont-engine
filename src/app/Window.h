#pragma once

#include <cstdint>
#include <functional>
#include <string>

struct GLFWwindow;

namespace pt::app {

class Window {
public:
    Window();
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Always created with no graphics API context attached -- backends
    // attach their own CAMetalLayer (Metal, Software-via-Metal-present)
    // or VkSurface (Vulkan via MoltenVK).
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

    // Engine-installed handler for hotkeys. Called for press events with
    // (key, mods) using the GLFW_KEY_* enums.  Esc-to-close is handled
    // before this hook fires.
    using KeyHandler = std::function<void(int key, int mods)>;
    void SetKeyHandler(KeyHandler h);

private:
    static void OnResize(GLFWwindow* w, int width, int height);
    static void OnKey(GLFWwindow* w, int key, int scancode, int action, int mods);

    GLFWwindow* handle_ = nullptr;
    int width_  = 0;
    int height_ = 0;
    std::string title_;
    KeyHandler  key_handler_;
};

}  // namespace pt::app

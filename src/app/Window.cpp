#include "Window.h"
#include "../core/Log.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace pt::app {

namespace {

bool g_glfw_inited = false;
int  g_window_count = 0;

void GlfwErrorCallback(int code, const char* desc) {
    LOG_ERROR("GLFW error {}: {}", code, desc ? desc : "(null)");
}

bool EnsureGlfw() {
    if (g_glfw_inited) return true;
    glfwSetErrorCallback(&GlfwErrorCallback);
    if (glfwInit() != GLFW_TRUE) {
        LOG_ERROR("glfwInit failed");
        return false;
    }
    g_glfw_inited = true;
    return true;
}

}  // namespace

Window::Window() = default;
Window::~Window() { Destroy(); }

bool Window::Create(int w, int h, std::string_view title, GraphicsApi api) {
    if (!EnsureGlfw()) return false;
    Destroy();
    title_.assign(title);
    api_ = api;

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE,    GLFW_TRUE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    if (api == GraphicsApi::OpenGL) {
        glfwWindowHint(GLFW_CLIENT_API,         GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,     GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    } else {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    handle_ = glfwCreateWindow(w, h, title_.c_str(), nullptr, nullptr);
    if (handle_ == nullptr) {
        LOG_ERROR("glfwCreateWindow({},{}) failed", w, h);
        return false;
    }
    width_  = w;
    height_ = h;
    glfwSetWindowUserPointer(handle_, this);
    glfwSetFramebufferSizeCallback(handle_, &Window::OnResize);
    glfwSetKeyCallback(handle_, &Window::OnKey);

    ++g_window_count;
    return true;
}

bool Window::Recreate(GraphicsApi api) {
    if (api == api_ && handle_ != nullptr) return true;
    int x = 0, y = 0;
    if (handle_ != nullptr) {
        glfwGetWindowPos(handle_, &x, &y);
    }
    int w = width_, h = height_;
    Destroy();
    if (!Create(w, h, title_, api)) return false;
    if (handle_ != nullptr) {
        glfwSetWindowPos(handle_, x, y);
    }
    return true;
}

void Window::Destroy() {
    if (handle_ != nullptr) {
        glfwDestroyWindow(handle_);
        handle_ = nullptr;
        if (--g_window_count == 0 && g_glfw_inited) {
            glfwTerminate();
            g_glfw_inited = false;
        }
    }
}

void Window::PollEvents() {
    if (g_glfw_inited) glfwPollEvents();
}

bool Window::ShouldClose() const {
    return handle_ != nullptr && glfwWindowShouldClose(handle_);
}

void Window::RequestClose() {
    if (handle_ != nullptr) glfwSetWindowShouldClose(handle_, GLFW_TRUE);
}

extern "C" void* pt_window_native_cocoa(GLFWwindow*);

void* Window::NativeHandle() const {
#if defined(__APPLE__)
    return pt_window_native_cocoa(handle_);
#else
    return nullptr;
#endif
}

void Window::OnResize(GLFWwindow* w, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self == nullptr) return;
    self->width_  = width;
    self->height_ = height;
}

void Window::SetKeyHandler(KeyHandler h) { key_handler_ = std::move(h); }

void Window::OnKey(GLFWwindow* w, int key, int /*scancode*/, int action, int mods) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(w, GLFW_TRUE);
        return;
    }
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self != nullptr && self->key_handler_) {
        self->key_handler_(key, mods);
    }
}

}  // namespace pt::app

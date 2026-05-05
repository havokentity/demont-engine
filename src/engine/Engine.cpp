#include "Engine.h"

#include "../app/ConsoleOverlay.h"
#include "../app/Window.h"
#include "../console/Console.h"
#include "../console/ConsoleServer.h"
#include "../core/Hardware/HardwareInfo.h"
#include "../core/Jobs/JobSystem.h"
#include "../core/Log.h"
#include "../core/Memory/Memory.h"
#include "../renderer/Camera.h"
#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <glm/glm.hpp>
#include <cstdint>

#include <fmt/format.h>

#include <chrono>
#include <cstdlib>
#include <thread>
#include <charconv>

namespace pt::engine {

namespace {

Engine* g_instance = nullptr;

// ---- CVars (registered statically) ----------------------------------------
namespace cvar {
    using namespace pt::console;
    PT_CVAR(net_port,         "27960", "WebSocket / HTTP port",          CVAR_ARCHIVE);
    PT_CVAR(net_line_port,    "27961", "Plain-text TCP port",            CVAR_ARCHIVE);
    PT_CVAR(net_bind_address, "127.0.0.1", "Bind address; loopback only by default", CVAR_ARCHIVE);
    PT_CVAR(app_window_width,  "1280", "Initial window width",           CVAR_ARCHIVE);
    PT_CVAR(app_window_height, "720",  "Initial window height",          CVAR_ARCHIVE);
    PT_CVAR(app_vsync,         "1",    "Swapchain vsync (1=on)",         CVAR_ARCHIVE);
    PT_CVAR(app_auto_open_console, "1",
            "Open the web console in the default browser at startup",     CVAR_ARCHIVE);
    PT_CVAR(app_overlay_enabled, "1",
            "Enable the in-window native console overlay (backtick toggles)", CVAR_ARCHIVE);
    PT_CVAR(r_backend,         "software", "One of none|software|metal|vulkan",CVAR_ARCHIVE);
    PT_CVAR(r_clear_color,     "0.18 0.05 0.28", "Background clear colour (R G B)", 0);
    PT_CVAR(r_mode,            "scene",
            "What to render: clear|scene (preview/raytrace/pathtrace land in P7)",
            CVAR_ARCHIVE);
    PT_CVAR(dev_cheats,        "0",    "Gate for CHEAT-flagged cvars",   0);
    PT_CVAR(dev_log_level,     "info", "error|warn|info|debug",          0);

    // Hardware info (filled at startup, READONLY).
    PT_CVAR(sys_cpu_model,    "?",  "CPU brand string",              CVAR_READONLY);
    PT_CVAR(sys_cpu_pcores,   "0",  "Performance cores",             CVAR_READONLY);
    PT_CVAR(sys_cpu_ecores,   "0",  "Efficiency cores",              CVAR_READONLY);
    PT_CVAR(sys_cpu_features, "?",  "SIMD features",                 CVAR_READONLY);
    PT_CVAR(sys_ram_total_mb, "0",  "Physical RAM in MB",            CVAR_READONLY);
    PT_CVAR(sys_os,           "?",  "Operating system",              CVAR_READONLY);
    PT_CVAR(sys_gpu_name,     "?",  "GPU model (filled by RHI)",     CVAR_READONLY);
    PT_CVAR(sys_gpu_unified,  "1",  "1 if unified memory architecture", CVAR_READONLY);
    PT_CVAR(sys_gpu_hwrt,     "0",  "1 if hardware ray tracing supported", CVAR_READONLY);
}  // namespace cvar

}  // namespace

Engine::Engine()  { g_instance = this; }
Engine::~Engine() { Shutdown(); if (g_instance == this) g_instance = nullptr; }

Engine* Engine::Instance() { return g_instance; }

bool Engine::Init() {
    pt::mem::Init();

    pt::hw::Populate();
    auto& hi = pt::hw::GetInfo();
    auto& C  = pt::console::Console::Get();
    C.SetCVarOverride("sys_cpu_model",    hi.cpu_model);
    C.SetCVarOverride("sys_cpu_pcores",   std::to_string(hi.cpu_pcores));
    C.SetCVarOverride("sys_cpu_ecores",   std::to_string(hi.cpu_ecores));
    C.SetCVarOverride("sys_cpu_features", hi.cpu_features);
    C.SetCVarOverride("sys_ram_total_mb", std::to_string(hi.ram_total_mb));
    C.SetCVarOverride("sys_os",           hi.os_name);

    RegisterCommands();

    // Job system.
    jobs_ = std::make_unique<pt::jobs::JobSystem>();
    jobs_->Init();
    pt::jobs::JobSystem::SetInstance(jobs_.get());

    // Window. Always created with no graphics-API context; each backend
    // attaches its own CAMetalLayer (or VkSurface for Vulkan) to the
    // NSWindow content view, so no recreate is needed on backend switch.
    auto* ww = C.FindCVar("app_window_width");
    auto* wh = C.FindCVar("app_window_height");
    int win_w = ww ? ww->GetInt() : 1280;
    int win_h = wh ? wh->GetInt() : 720;
    window_ = std::make_unique<pt::app::Window>();
    if (!window_->Create(win_w, win_h, "PathTracer")) {
        LOG_ERROR("Failed to open window");
        return false;
    }

    // Console server.
    server_ = std::make_unique<pt::console::ConsoleServer>();
    pt::console::ConsoleServer::SetGlobalInstance(server_.get());
    pt::log::AddSink(&pt::console::ConsoleServer::OnLog);

    pt::console::ConsoleServer::Config sc;
    if (auto* p = C.FindCVar("net_port"))         sc.http_port = static_cast<std::uint16_t>(p->GetInt());
    if (auto* p = C.FindCVar("net_line_port"))    sc.line_port = static_cast<std::uint16_t>(p->GetInt());
    if (auto* p = C.FindCVar("net_bind_address")) sc.bind_address = p->value;
    server_->Start(sc, &C);

    // Boot the requested backend so the window renders something on
    // startup (defaults to "software" -- the only one online today).
    if (auto* v = C.FindCVar("r_backend")) {
        BackendType t = BackendType::None;
        if      (v->value == "software") t = BackendType::Software;
        else if (v->value == "metal")    t = BackendType::Metal;
        else if (v->value == "vulkan")   t = BackendType::Vulkan;
        if (t != BackendType::None) RequestBackendSwitch(t);
    }

    // Native in-window console overlay (uses macOS AppKit -- not a GUI
    // library; AppKit ships with the OS). Backtick toggles it.
    if (window_) {
        if (auto* ov = C.FindCVar("app_overlay_enabled"); ov && ov->GetBool()) {
            overlay_ = std::make_unique<pt::app::ConsoleOverlay>();
            if (overlay_->Init(window_->NativeHandle())) {
                pt::log::AddSink(&pt::app::ConsoleOverlay::OnLog);
            } else {
                overlay_.reset();
            }
        }

        window_->SetKeyHandler([this](int key, int /*mods*/) {
            constexpr int kGrave = 96;  // GLFW_KEY_GRAVE_ACCENT
            if (key != kGrave) return;
            if (overlay_) {
                overlay_->Toggle();
            } else {
                OpenWebConsole();
            }
        });
    }

    if (auto* v = C.FindCVar("app_auto_open_console"); v && v->GetBool()) {
        OpenWebConsole();
    }

    LOG_INFO("Engine initialized.");
    return true;
}

void Engine::OpenWebConsole() {
    auto* C  = &pt::console::Console::Get();
    auto* p  = C->FindCVar("net_port");
    auto* a  = C->FindCVar("net_bind_address");
    int  port = p ? p->GetInt() : 27960;
    std::string host = (a && !a->value.empty() && a->value != "0.0.0.0")
                       ? a->value : std::string("localhost");
    auto url = fmt::format("http://{}:{}/", host, port);
    LOG_INFO("Opening console: {}", url);

#if defined(__APPLE__)
    auto cmd = fmt::format("/usr/bin/open '{}' >/dev/null 2>&1", url);
    int rc = std::system(cmd.c_str());
    if (rc != 0) LOG_WARN("`open` returned {} -- visit {} manually", rc, url);
#elif defined(_WIN32)
    auto cmd = fmt::format("cmd /c start \"\" \"{}\"", url);
    std::system(cmd.c_str());
#else
    auto cmd = fmt::format("xdg-open '{}' >/dev/null 2>&1 &", url);
    std::system(cmd.c_str());
#endif
}

void Engine::Shutdown() {
    TearDownDevice();

    pt::log::RemoveAllSinks();
    if (overlay_) overlay_->Shutdown();
    overlay_.reset();
    if (server_) server_->Stop();
    server_.reset();
    pt::console::ConsoleServer::SetGlobalInstance(nullptr);

    if (window_) window_->Destroy();
    window_.reset();

    if (jobs_) jobs_->Shutdown();
    pt::jobs::JobSystem::SetInstance(nullptr);
    jobs_.reset();
}

void Engine::TearDownDevice() {
    if (device_) {
        // Final frame clears the drawable to black so the window doesn't
        // freeze on whatever colour was last presented (the layer keeps
        // showing its last contents until something else writes to it).
        if (clear_pipeline_id_ != 0) {
            auto fc = device_->BeginFrame();
            if (auto* cb = device_->AcquireCommandBuffer()) {
                cb->BindComputePipeline(pt::rhi::PipelineHandle{clear_pipeline_id_});
                cb->BindStorageTexture(0, fc.swapchain_image);
                const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                cb->PushConstants(black, sizeof(black));
                cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);
                device_->Submit(cb);
                device_->EndFrame(cb);
            }
        }
        device_->WaitIdle();
        device_.reset();
    }
    clear_pipeline_id_ = 0;
}

void Engine::RequestBackendSwitch(BackendType to) {
    if (to == current_backend_ && (to == BackendType::None || device_)) return;
    LOG_INFO("backend switch: {} -> {}",
             pt::rhi::BackendName(current_backend_),
             pt::rhi::BackendName(to));

    TearDownDevice();

    if (to == BackendType::None) {
        current_backend_ = BackendType::None;
        return;
    }

    // No window recreate needed: window is always NO_API and each backend
    // attaches its own surface (CAMetalLayer / VkSurface) to it.

    pt::rhi::NativeWindowHandle nw{
        .opaque = window_ ? static_cast<void*>(window_->Handle()) : nullptr,
        .width  = window_ ? window_->Width()  : 0,
        .height = window_ ? window_->Height() : 0,
    };
    device_ = pt::rhi::Device::Create(to, nw);
    if (!device_) {
        LOG_ERROR("failed to create {} device", pt::rhi::BackendName(to));
        current_backend_ = BackendType::None;
        return;
    }

    current_backend_ = to;
    pt::console::Console::Get().SetCVarOverride(
        "sys_gpu_name", device_->DeviceName());
    pt::console::Console::Get().SetCVarOverride(
        "sys_gpu_hwrt", device_->SupportsHardwareRT() ? "1" : "0");

    // Build per-backend pipelines.  Software backend tracks them by name
    // only (CPU fallback in P5+); Metal returns the precompiled Slang->MSL
    // PSO.  Both kernels expose `main` and route through the same
    // CreateComputePipeline interface.
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "clear", .bytecode = {}, .debug_name = "clear",
        };
        clear_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "scene", .bytecode = {}, .debug_name = "scene",
        };
        scene_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }
}

namespace {

// Parse a "R G B" cvar into a float3 (clamped to 0..1 for safety).
glm::vec3 ParseRGB(std::string_view sv, glm::vec3 fallback) {
    glm::vec3 c = fallback;
    std::size_t i = 0;
    for (int k = 0; k < 3; ++k) {
        while (i < sv.size() && sv[i] == ' ') ++i;
        if (i >= sv.size()) return c;
        char* end = nullptr;
        c[k] = std::strtof(sv.data() + i, &end);
        if (end == sv.data() + i) return c;
        i = static_cast<std::size_t>(end - sv.data());
    }
    return glm::clamp(c, 0.0f, 1.0f);
}

// Pack an rgb triple in 0..1 into the low 24 bits of a float (raw bits).
float PackBgRGB(glm::vec3 c) {
    auto r = static_cast<std::uint32_t>(glm::clamp(c.r, 0.0f, 1.0f) * 255.0f + 0.5f);
    auto g = static_cast<std::uint32_t>(glm::clamp(c.g, 0.0f, 1.0f) * 255.0f + 0.5f);
    auto b = static_cast<std::uint32_t>(glm::clamp(c.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    std::uint32_t packed = (b << 16) | (g << 8) | r;
    float out;
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::memcpy(&out, &packed, sizeof(out));
    return out;
}

}  // namespace

void Engine::RenderFrame() {
    if (!device_) return;

    auto& C = pt::console::Console::Get();
    bool scene_mode = false;
    if (auto* v = C.FindCVar("r_mode")) scene_mode = (v->value == "scene");

    std::uint64_t pipeline = scene_mode ? scene_pipeline_id_ : clear_pipeline_id_;
    if (pipeline == 0) {
        // Fall back to clear if the requested pipeline isn't built (e.g.
        // software backend doesn't ship a scene shader yet).
        pipeline = clear_pipeline_id_;
    }
    if (pipeline == 0) return;

    auto fc = device_->BeginFrame();
    auto* cb = device_->AcquireCommandBuffer();
    cb->BindComputePipeline(pt::rhi::PipelineHandle{pipeline});
    cb->BindStorageTexture(0, fc.swapchain_image);

    glm::vec3 bg = ParseRGB(C.FindCVar("r_clear_color") ? C.FindCVar("r_clear_color")->value
                                                        : std::string_view{},
                            glm::vec3{0.18f, 0.05f, 0.28f});

    if (pipeline == clear_pipeline_id_) {
        // Clear path: 16-byte push of the colour.
        const float color[4] = {bg.r, bg.g, bg.b, 1.0f};
        cb->PushConstants(color, sizeof(color));
    } else {
        // Scene path: 64-byte push containing the camera basis + sky.
        // Camera is hardcoded for now -- WASD / mouse-look land in the
        // rest of P5.
        pt::renderer::Camera cam;   // defaults: pos=(0,1.5,4), pitch=-0.2
        const auto fwd   = cam.Forward();
        const auto right = cam.Right();
        const auto up    = cam.Up();
        const float aspect = (fc.height > 0)
                               ? float(fc.width) / float(fc.height) : 1.0f;
        struct Push {
            float pos_fovtan[4];
            float fwd_aspect[4];
            float right_xyz[4];
            float up_bg_pack[4];
        } push{};
        push.pos_fovtan[0] = cam.pos.x;
        push.pos_fovtan[1] = cam.pos.y;
        push.pos_fovtan[2] = cam.pos.z;
        push.pos_fovtan[3] = cam.FovYTan();
        push.fwd_aspect[0] = fwd.x;
        push.fwd_aspect[1] = fwd.y;
        push.fwd_aspect[2] = fwd.z;
        push.fwd_aspect[3] = aspect;
        push.right_xyz[0]  = right.x;
        push.right_xyz[1]  = right.y;
        push.right_xyz[2]  = right.z;
        push.right_xyz[3]  = 0.0f;
        push.up_bg_pack[0] = up.x;
        push.up_bg_pack[1] = up.y;
        push.up_bg_pack[2] = up.z;
        push.up_bg_pack[3] = PackBgRGB(bg);
        static_assert(sizeof(push) == 64, "push must fit in 64 bytes");
        cb->PushConstants(&push, sizeof(push));
    }

    auto wg_x = (fc.width  + 7) / 8;
    auto wg_y = (fc.height + 7) / 8;
    cb->Dispatch(wg_x, wg_y, 1);

    device_->Submit(cb);
    device_->EndFrame(cb);
}

void Engine::Tick(double dt) {
    pt::console::Console::Get().Drain();

    auto t0 = std::chrono::steady_clock::now();
    RenderFrame();
    auto frame_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // Broadcast frame_stats throttled to ~10 Hz so the network doesn't
    // get spammed at 240+ fps.
    static double accum_s = 0.0;
    static int    accum_frames = 0;
    static double accum_render_ms = 0.0;
    accum_s         += dt;
    accum_frames    += 1;
    accum_render_ms += frame_ms;
    if (accum_s >= 0.1 && server_) {
        double fps = accum_frames / accum_s;
        double avg_render = accum_render_ms / accum_frames;
        int w = window_ ? window_->Width()  : 0;
        int h = window_ ? window_->Height() : 0;
        auto data = fmt::format(
            R"({{"fps":{:.1f},"frame_ms":{:.3f},"trace_ms":{:.3f},"backend":"{}","resolution":[{},{}]}})",
            fps, accum_render_ms / accum_frames, avg_render,
            pt::rhi::BackendName(current_backend_), w, h);
        server_->BroadcastEvent("frame_stats", data);
        accum_s = 0.0;
        accum_frames = 0;
        accum_render_ms = 0.0;
    }
}

void Engine::Run() {
    using clk = std::chrono::steady_clock;
    auto last = clk::now();
    while (!wants_quit_ && !window_->ShouldClose()) {
        auto now = clk::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;

        window_->PollEvents();
        Tick(dt);

        // If no device is bound yet we'd busy-loop; throttle.  With a
        // device, the swapchain present already paces us via vsync.
        if (!device_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }
    LOG_INFO("Run loop exited.");
}

// ----- Commands -------------------------------------------------------------

void Engine::RegisterCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("quit", "Exit the application cleanly.",
        [this](auto, pt::console::Output& out) {
            out.PrintLine("bye.");
            wants_quit_ = true;
            if (window_) window_->RequestClose();
        });

    C.RegisterCommand("echo", "Print arguments back.",
        [](auto args, pt::console::Output& out) {
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (i) out.Print(" ");
                out.Print(args[i]);
            }
            out.Print("\n");
        });

    C.RegisterCommand("sys_info", "Summarize hardware.",
        [](auto, pt::console::Output& out) {
            auto& C2 = pt::console::Console::Get();
            auto get = [&](const char* n)->std::string {
                auto* v = C2.FindCVar(n); return v ? v->value : std::string("?");
            };
            out.FormatLine("CPU      {} ({}P + {}E)", get("sys_cpu_model"),
                           get("sys_cpu_pcores"), get("sys_cpu_ecores"));
            out.FormatLine("Features {}", get("sys_cpu_features"));
            out.FormatLine("RAM      {} MB", get("sys_ram_total_mb"));
            out.FormatLine("OS       {}", get("sys_os"));
            out.FormatLine("GPU      {}", get("sys_gpu_name"));
        });

    C.RegisterCommand("mem_report", "Per-tag memory breakdown.",
        [](auto, pt::console::Output& out) {
            pt::mem::TagStats s[pt::kMemTagCount];
            pt::mem::GetReport(s);
            out.FormatLine("{:<12} {:>14} {:>14} {:>10} {:>10}",
                           "tag", "live(B)", "peak(B)", "allocs", "frees");
            std::size_t tl = 0, tp = 0;
            for (int i = 0; i < pt::kMemTagCount; ++i) {
                out.FormatLine("{:<12} {:>14} {:>14} {:>10} {:>10}",
                               pt::MemTagName(static_cast<pt::MemTag>(i)),
                               s[i].live_bytes, s[i].peak_bytes,
                               s[i].alloc_count, s[i].free_count);
                tl += s[i].live_bytes;
                tp += s[i].peak_bytes;
            }
            out.FormatLine("{:<12} {:>14} {:>14}", "TOTAL", tl, tp);
        });

    C.RegisterCommand("list_cvars", "List cvars (optional prefix).",
        [](auto args, pt::console::Output& out) {
            std::string prefix;
            if (!args.empty()) prefix.assign(args[0]);
            pt::console::Console::Get().EnumerateCVars(prefix,
                [&](pt::console::CVar& v) {
                    out.FormatLine("{} = \"{}\" ({})", v.name, v.value, v.description);
                });
        });

    C.RegisterCommand("list_commands", "List commands (optional prefix).",
        [](auto args, pt::console::Output& out) {
            std::string prefix;
            if (!args.empty()) prefix.assign(args[0]);
            pt::console::Console::Get().EnumerateCommands(prefix,
                [&](pt::console::Command& c) {
                    out.FormatLine("{} -- {}", c.name, c.description);
                });
        });

    C.RegisterCommand("exec", "Run a .cfg script: exec autoexec.cfg",
        [](auto args, pt::console::Output& out) {
            if (args.empty()) {
                out.PrintLine("usage: exec <file>");
                return;
            }
            std::string path(args[0]);
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) { out.FormatLine("cannot open: {}", path); return; }
            std::string body;
            char buf[4096];
            while (auto n = std::fread(buf, 1, sizeof(buf), f)) body.append(buf, n);
            std::fclose(f);
            auto r = pt::console::Console::Get().ExecuteScript(body);
            if (!r.output.empty()) out.Print(r.output);
            if (!r.ok) out.FormatLine("exec error: {}", r.error);
        });

    // r_backend: validate against the known set + react to changes.
    if (auto* v = C.FindCVar("r_backend")) {
        v->allowed_values = {"none", "software", "metal", "vulkan"};
        v->on_change = [this](const pt::console::CVar& cv) {
            BackendType t = BackendType::None;
            if      (cv.value == "software") t = BackendType::Software;
            else if (cv.value == "metal")    t = BackendType::Metal;
            else if (cv.value == "vulkan")   t = BackendType::Vulkan;
            RequestBackendSwitch(t);
        };
    }
    // r_mode: clear|scene for now (preview/raytrace/pathtrace land in P7).
    if (auto* v = C.FindCVar("r_mode")) {
        v->allowed_values = {"clear", "scene"};
    }
    // dev_log_level: validate
    if (auto* v = C.FindCVar("dev_log_level")) {
        v->allowed_values = {"error", "warn", "info", "debug"};
    }
    // app_vsync / app_overlay_enabled / app_auto_open_console / dev_cheats:
    // boolean toggles -- accept 0|1.
    for (const char* n : {"app_vsync", "app_overlay_enabled",
                          "app_auto_open_console", "dev_cheats"}) {
        if (auto* v = C.FindCVar(n)) v->allowed_values = {"0", "1"};
    }
}

}  // namespace pt::engine

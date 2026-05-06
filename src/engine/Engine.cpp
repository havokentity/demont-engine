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
#include "../renderer/Csg/CsgScene.h"
#include "../renderer/HdrImage.h"
#include "../renderer/MeshGen.h"
#include "../rhi/CommandBuffer.h"
#include "../rhi/Device.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define TOML_EXCEPTIONS 0
#include <toml++/toml.h>
#include <cstdint>

#include <fmt/format.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <thread>

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
    PT_CVAR(app_auto_open_console, "0",
            "Open the web console in the default browser at startup. Off by default; use `web_console` command to open on demand.",
            CVAR_ARCHIVE);
    PT_CVAR(app_overlay_enabled, "1",
            "Enable the in-window native console overlay (backtick toggles)", CVAR_ARCHIVE);
    PT_CVAR(r_theme, "hardcore",
            "Web console theme: hardcore|amber|synthwave|matrix|vault|sakura|mono",
            CVAR_ARCHIVE);
    PT_CVAR(r_backend,         "metal",    "One of none|software|metal|vulkan",CVAR_ARCHIVE);
    PT_CVAR(r_max_bounces,     "8",  "Max path bounces per ray",          CVAR_ARCHIVE);
    PT_CVAR(r_spp,             "1",  "Samples per pixel per dispatch (>=1). Higher = cleaner motion frames at proportional GPU cost.", CVAR_ARCHIVE);
    PT_CVAR(r_denoiser,        "off","Denoiser: off|metalfx (Mac MetalFX TemporalDenoisedScaler).", CVAR_ARCHIVE);
    PT_CVAR(r_env_map,         "",   "Path to a Radiance .hdr environment map. Empty = procedural gradient sky.", CVAR_ARCHIVE);
    PT_CVAR(r_env_intensity,   "1.0","Scalar multiplier on env-map samples. Useful for darkening/brightening the IBL without re-authoring the HDRI.", CVAR_ARCHIVE);

    // Camera controls.  Mouse-look engages while RIGHT mouse is held.
    PT_CVAR(cam_speed,         "3.0", "Movement speed (units/sec)",        CVAR_ARCHIVE);
    PT_CVAR(cam_sprint_mult,   "3.0", "Speed multiplier with shift",       CVAR_ARCHIVE);
    PT_CVAR(cam_sensitivity,   "0.12","Mouse-look sensitivity (deg/pixel)", CVAR_ARCHIVE);
    PT_CVAR(cam_fov,           "60.0","Vertical field of view (degrees)",  CVAR_ARCHIVE);
    PT_CVAR(cam_pos,           "0 1.5 4", "Camera position (x y z)",       CVAR_ARCHIVE);
    PT_CVAR(cam_yaw,           "0",       "Yaw in degrees",                CVAR_ARCHIVE);
    PT_CVAR(cam_pitch,         "-11.5",   "Pitch in degrees (clamped +/- 85)", CVAR_ARCHIVE);
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

    camera_ = std::make_unique<pt::renderer::Camera>();

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

    // P11 persistence: replay last session's archived cvars first, then
    // the user's autoexec.cfg. Both are exec'd as plain console scripts
    // so any command (not just `<cvar> <value>`) is valid in autoexec.
    auto exec_if_exists = [](const char* path) {
        FILE* f = std::fopen(path, "rb");
        if (f == nullptr) return;
        std::string body; char buf[4096];
        while (auto n = std::fread(buf, 1, sizeof(buf), f)) body.append(buf, n);
        std::fclose(f);
        auto r = pt::console::Console::Get().ExecuteScript(body);
        if (!r.ok) LOG_WARN("{}: {}", path, r.error);
        LOG_INFO("loaded {}", path);
    };
    exec_if_exists("demont.cfg");      // archived cvars from last quit
    exec_if_exists("autoexec.cfg");    // user-supplied startup script (overrides above)

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
    if (!window_->Create(win_w, win_h, "DeMonT Engine")) {
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

    // CSG scene -- the source of truth for triangle-mesh geometry from
    // P9 onward. Seeded with the headline drilled-cube scene so first-
    // frame renders something interesting.
    csg_scene_ = std::make_unique<pt::csg::CsgScene>();
    SeedDefaultCsgScene();

    // Analytic primitives -- spheres + planes that the unified renderer
    // intersects alongside the mesh TLAS. Seeded with the canonical
    // 3-sphere + ground-plane scene.
    SeedDefaultPrimitives();

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

    // Seed camera from cvars (so cam_pos / cam_yaw / cam_pitch persist
    // across runs once we save archived cvars in P11).
    if (auto* v = C.FindCVar("cam_pos")) {
        std::string_view sv = v->value;
        std::size_t i = 0;
        glm::vec3 p = camera_->pos;
        for (int k = 0; k < 3; ++k) {
            while (i < sv.size() && sv[i] == ' ') ++i;
            char* end = nullptr;
            p[k] = std::strtof(sv.data() + i, &end);
            if (!end || end == sv.data() + i) break;
            i = static_cast<std::size_t>(end - sv.data());
        }
        camera_->pos = p;
    }
    if (auto* v = C.FindCVar("cam_yaw"))   camera_->yaw   = glm::radians(v->GetFloat());
    if (auto* v = C.FindCVar("cam_pitch")) camera_->pitch = glm::radians(v->GetFloat());
    if (auto* v = C.FindCVar("cam_fov"))   camera_->fov_deg = v->GetFloat();
    camera_->ClampPitch();

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

    // P11 persistence: dump every CVAR_ARCHIVE cvar that's been changed
    // away from its default into demont.cfg so the next launch picks
    // back up the user's settings.
    if (int n = pt::console::Console::Get().SaveArchivedCvars("demont.cfg"); n >= 0) {
        LOG_INFO("saved {} archived cvar(s) to demont.cfg", n);
    } else {
        LOG_WARN("could not write demont.cfg");
    }

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
    // Drain any in-flight CSG bake so its worker doesn't touch the
    // engine after the device dies.
    if (jobs_ && bake_handle_.internal != nullptr) {
        jobs_->Wait(bake_handle_);
        bake_handle_ = {};
    }
    bake_phase_.store(0, std::memory_order_release);
    pending_baked_.reset();

    accum_texture_id_ = 0;
    accum_w_ = accum_h_ = 0;
    if (device_) {
        if (scene_tlas_id_        != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
        if (box_blas_id_          != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
        if (box_vbuf_id_          != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
        if (box_ibuf_id_          != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
        if (prim_buffer_id_       != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{prim_buffer_id_});
        if (denoise_color_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        if (depth_tex_id_         != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
        if (motion_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
        if (env_map_tex_id_       != 0) device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
    }
    scene_tlas_id_        = 0;
    box_blas_id_          = 0;
    box_vbuf_id_          = 0;
    box_ibuf_id_          = 0;
    prim_buffer_id_       = 0;
    prim_buffer_capacity_ = 0;
    denoise_color_tex_id_ = 0;
    depth_tex_id_         = 0;
    motion_tex_id_        = 0;
    env_map_tex_id_       = 0;
    denoiser_active_      = false;
    prev_view_proj_valid_ = false;
    primitives_dirty_     = true;        // re-upload on next device

    if (device_) {
        device_->WaitIdle();
        device_.reset();
    }
    pathtrace_pipeline_id_ = 0;
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

    // The unified path-tracer pipeline is the only kernel from here on.
    // Both Metal and Vulkan backends expose it under the name "pathtrace".
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "pathtrace", .bytecode = {}, .debug_name = "pathtrace",
        };
        pathtrace_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }

    // Mark the analytic-primitive buffer dirty so it gets uploaded on
    // the first frame against this device. The mesh BLAS + TLAS are
    // driven by CsgScene -- EnsureMeshUpdated() handles the bake job.
    primitives_dirty_ = true;

    // Reload env map on the new device so its texture handle is valid.
    if (auto* v = pt::console::Console::Get().FindCVar("r_env_map");
        v && !v->value.empty()) {
        ReloadEnvMap(v->value);
    }
}

void Engine::SeedDefaultCsgScene() {
    if (!csg_scene_) return;
    csg_scene_->Reset();                                 // unit box at id 1, root
    csg_scene_->AddSphere(2, 0.6f, 48, 0.0f, 0.5f, 0.0f); // sphere at box centre
    csg_scene_->Combine(3, pt::csg::OpType::Subtract, 1, 2); // box - sphere
    csg_scene_->SetRoot(3);
}

void Engine::EnsureMeshUpdated() {
    if (!device_ || current_backend_ != BackendType::Metal) return;
    if (!csg_scene_) return;

    // Phase 2: the worker has a result waiting. Pull it onto the main
    // thread, free the job handle, swap in fresh GPU resources.
    if (bake_phase_.load(std::memory_order_acquire) == 2) {
        if (jobs_ && bake_handle_.internal != nullptr) {
            jobs_->Wait(bake_handle_);
        }
        bake_handle_ = {};
        std::unique_ptr<pt::csg::BakedMesh> baked = std::move(pending_baked_);
        bake_phase_.store(0, std::memory_order_release);
        if (baked && !baked->Empty()) {
            RebuildMeshResources(*baked);
            accum_dirty_ = true;
        } else {
            // Bake produced empty geometry (degenerate CSG). Drop any
            // existing mesh resources so r_mode mesh falls back to clear
            // rather than rendering a stale shape.
            if (scene_tlas_id_ != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
            if (box_blas_id_   != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
            if (box_vbuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
            if (box_ibuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
            scene_tlas_id_ = box_blas_id_ = box_vbuf_id_ = box_ibuf_id_ = 0;
        }
    }

    // Phase 0: idle. If the scene is dirty, kick a fresh bake. Ack the
    // scene clean BEFORE submitting so a mutation during the bake re-marks
    // it dirty -- we'll then bake again on the next ready->idle transition.
    if (bake_phase_.load(std::memory_order_acquire) == 0 &&
        csg_scene_->Dirty() && jobs_) {
        csg_scene_->AcknowledgeClean();
        bake_phase_.store(1, std::memory_order_release);
        bake_handle_ = jobs_->Submit([this] {
            std::string err;
            auto baked = std::make_unique<pt::csg::BakedMesh>(
                csg_scene_->Bake(&err));
            if (baked->Empty() && !err.empty()) {
                LOG_WARN("CSG bake: {}", err);
            }
            pending_baked_ = std::move(baked);
            bake_phase_.store(2, std::memory_order_release);
        });
    }
}

void Engine::RebuildMeshResources(const pt::csg::BakedMesh& baked) {
    if (!device_) return;

    // Drain any in-flight GPU work so it's safe to destroy old resources.
    device_->WaitIdle();

    if (scene_tlas_id_ != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{scene_tlas_id_});
    if (box_blas_id_   != 0) device_->DestroyAccelStruct(pt::rhi::AccelStructHandle{box_blas_id_});
    if (box_vbuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_vbuf_id_});
    if (box_ibuf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{box_ibuf_id_});
    scene_tlas_id_ = box_blas_id_ = box_vbuf_id_ = box_ibuf_id_ = 0;

    const std::size_t vbytes = sizeof(float) * baked.positions.size();
    const std::size_t ibytes = sizeof(std::uint32_t) * baked.indices.size();
    auto vbuf = device_->CreateBuffer({
        .size = vbytes, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "csg_vbuf",
    });
    auto ibuf = device_->CreateBuffer({
        .size = ibytes, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "csg_ibuf",
    });
    if (vbuf.id == 0 || ibuf.id == 0) {
        LOG_ERROR("CSG: failed to allocate vertex/index storage buffers");
        return;
    }
    device_->WriteBuffer(vbuf, baked.positions.data(), vbytes);
    device_->WriteBuffer(ibuf, baked.indices.data(),   ibytes);
    box_vbuf_id_ = vbuf.id;
    box_ibuf_id_ = ibuf.id;

    pt::rhi::BLASDesc blas_desc{
        .vertex_positions = baked.positions.data(),
        .vertex_count     = baked.VertexCount(),
        .indices          = baked.indices.data(),
        .index_count      = static_cast<std::uint32_t>(baked.indices.size()),
        .debug_name       = "csg",
    };
    auto blas = device_->CreateBLAS(blas_desc);
    if (blas.id == 0) {
        LOG_ERROR("CSG: BLAS build failed ({} verts, {} tris)",
                  baked.VertexCount(), baked.TriangleCount());
        return;
    }
    box_blas_id_ = blas.id;

    // Single instance, identity transform -- the CSG bake already places
    // geometry in world space.
    pt::rhi::TLASInstance inst{};
    inst.blas = blas;
    inst.transform[0] = 1; inst.transform[1] = 0; inst.transform[2] = 0; inst.transform[3]  = 0.0f;
    inst.transform[4] = 0; inst.transform[5] = 1; inst.transform[6] = 0; inst.transform[7]  = 0.0f;
    inst.transform[8] = 0; inst.transform[9] = 0; inst.transform[10]= 1; inst.transform[11] = 0.0f;
    inst.instance_id = 0;
    inst.mask        = 0xFF;
    std::vector<pt::rhi::TLASInstance> insts{inst};
    pt::rhi::TLASDesc tlas_desc{ .instances = insts, .debug_name = "scene" };
    auto tlas = device_->CreateTLAS(tlas_desc);
    scene_tlas_id_ = tlas.id;
    if (scene_tlas_id_ == 0) {
        LOG_ERROR("CSG: TLAS build failed");
        return;
    }
    LOG_INFO("CSG: rebuilt mesh ({} verts, {} tris)",
             baked.VertexCount(), baked.TriangleCount());
}

void Engine::SeedDefaultPrimitives() {
    primitives_.clear();
    auto rgb = [](float r, float g, float b) {
        return std::array<float, 3>{r, g, b};
    };
    auto add_sphere = [&](std::uint32_t id, float x, float y, float z, float r,
                          AnalyticPrim::Material mat,
                          std::array<float, 3> color, float roughness, float ior) {
        AnalyticPrim p{};
        p.type        = AnalyticPrim::Sphere;
        p.material    = mat;
        p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
        p.radius_or_d = r;
        p.albedo[0]   = color[0]; p.albedo[1] = color[1]; p.albedo[2] = color[2];
        p.roughness   = roughness;
        p.ior         = ior;
        primitives_[id] = p;
    };
    auto add_plane = [&](std::uint32_t id, float nx, float ny, float nz, float d,
                         AnalyticPrim::Material mat, std::array<float, 3> color) {
        AnalyticPrim p{};
        p.type        = AnalyticPrim::Plane;
        p.material    = mat;
        p.pos_or_n[0] = nx; p.pos_or_n[1] = ny; p.pos_or_n[2] = nz;
        p.radius_or_d = d;
        p.albedo[0]   = color[0]; p.albedo[1] = color[1]; p.albedo[2] = color[2];
        p.roughness   = 0.0f;
        p.ior         = 1.0f;
        primitives_[id] = p;
    };
    add_sphere(1, -1.2f, 0.5f,  0.2f, 0.5f, AnalyticPrim::Lambert,    rgb(0.92f, 0.30f, 0.30f), 0.0f,  1.0f);
    add_sphere(2,  0.0f, 0.5f,  0.0f, 0.5f, AnalyticPrim::Metal,      rgb(1.00f, 0.85f, 0.45f), 0.05f, 1.0f);
    add_sphere(3,  1.2f, 0.5f, -0.2f, 0.5f, AnalyticPrim::Dielectric, rgb(1.00f, 1.00f, 1.00f), 0.0f,  1.5f);
    add_plane (4,  0.0f, 1.0f,  0.0f, 0.0f, AnalyticPrim::Lambert,    rgb(0.55f, 0.55f, 0.55f));
    primitives_dirty_ = true;
    accum_dirty_      = true;
}

void Engine::ReloadEnvMap(const std::string& path) {
    if (!device_) {
        // Defer: cvar set before backend is up. Stash the path and apply
        // on the next RequestBackendSwitch.
        env_map_path_ = path;
        return;
    }

    if (env_map_tex_id_ != 0) {
        device_->WaitIdle();
        device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
        env_map_tex_id_ = 0;
    }
    env_map_path_ = path;
    if (path.empty()) {
        accum_dirty_ = true;     // sky changed -> pixel values change
        return;
    }

    std::string err;
    auto img = pt::renderer::LoadRadianceHdr(path, &err);
    if (img.Empty()) {
        LOG_WARN("env_map: failed to load '{}': {}", path, err);
        accum_dirty_ = true;
        return;
    }

    auto h = device_->CreateTexture({
        .width  = img.width,
        .height = img.height,
        .format = pt::rhi::TextureFormat::RGBA32F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "env_map",
    });
    if (h.id == 0) {
        LOG_ERROR("env_map: CreateTexture({}x{}) failed", img.width, img.height);
        return;
    }
    // Repack RGB -> RGBA (the texture format is 4-channel; alpha unused).
    std::vector<float> rgba(std::size_t(img.width) * img.height * 4);
    for (std::size_t i = 0; i < std::size_t(img.width) * img.height; ++i) {
        rgba[i * 4 + 0] = img.rgb[i * 3 + 0];
        rgba[i * 4 + 1] = img.rgb[i * 3 + 1];
        rgba[i * 4 + 2] = img.rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0f;
    }
    // No WriteTexture in the RHI yet -- piggyback on a temp staging buffer
    // copy via the buffer-to-texture path. For now, we have no such path
    // for textures either, so reuse the same trick the accum_hdr does:
    // since CreateTexture allocates shared storage on Apple Silicon, we
    // can use the read-back path's inverse if it existed. Simpler: add
    // a dedicated upload via the device.
    // TODO(rhi): proper UploadTexture() in the RHI; for now use the
    // backend-specific raw write via a one-off blit through a staging
    // buffer. We'll do this inline as a follow-up. For the moment,
    // upload via the existing WriteBuffer path is not applicable to
    // textures, so we must add a small helper. Doing it via a dedicated
    // call below.
    if (!device_->WriteTexture(h, rgba.data(), rgba.size() * sizeof(float))) {
        LOG_ERROR("env_map: WriteTexture failed");
        device_->DestroyTexture(h);
        return;
    }
    env_map_tex_id_ = h.id;
    accum_dirty_ = true;
    LOG_INFO("env_map: loaded {} ({}x{} HDR)", path, img.width, img.height);
}

void Engine::EnsurePrimitivesUploaded() {
    if (!device_) return;
    if (!primitives_dirty_) return;

    const std::uint32_t count = static_cast<std::uint32_t>(primitives_.size());
    constexpr std::uint32_t kFloatsPerPrim = 12;   // 3 float4s
    constexpr std::uint32_t kBytesPerPrim  = sizeof(float) * kFloatsPerPrim;

    // Allocate / grow the storage buffer when needed. We grow by powers
    // of two from a 16-prim floor so steady-state edits don't reallocate.
    if (count > prim_buffer_capacity_) {
        std::uint32_t new_cap = prim_buffer_capacity_ ? prim_buffer_capacity_ : 16u;
        while (new_cap < count) new_cap *= 2u;
        if (prim_buffer_id_ != 0) {
            device_->WaitIdle();
            device_->DestroyBuffer(pt::rhi::BufferHandle{prim_buffer_id_});
            prim_buffer_id_ = 0;
        }
        auto buf = device_->CreateBuffer({
            .size = std::size_t(new_cap) * kBytesPerPrim,
            .usage = pt::rhi::BufferUsage::Storage,
            .debug_name = "analytic_primitives",
        });
        if (buf.id == 0) {
            LOG_ERROR("primitive buffer allocation failed (capacity {})", new_cap);
            return;
        }
        prim_buffer_id_       = buf.id;
        prim_buffer_capacity_ = new_cap;
    }

    if (count == 0) {
        primitives_dirty_ = false;
        accum_dirty_      = true;
        return;
    }

    // Pack into a temporary float vector matching the shader layout:
    //   v0 = (xyz: pos/normal,  w: radius/d)
    //   v1 = (rgb: albedo,      a: roughness)
    //   v2 = (x: type,  y: mat, z: ior, w: pad)
    std::vector<float> packed(std::size_t(count) * kFloatsPerPrim, 0.0f);
    std::size_t offset = 0;
    for (const auto& [id, p] : primitives_) {
        packed[offset + 0] = p.pos_or_n[0];
        packed[offset + 1] = p.pos_or_n[1];
        packed[offset + 2] = p.pos_or_n[2];
        packed[offset + 3] = p.radius_or_d;
        packed[offset + 4] = p.albedo[0];
        packed[offset + 5] = p.albedo[1];
        packed[offset + 6] = p.albedo[2];
        packed[offset + 7] = p.roughness;
        packed[offset + 8]  = static_cast<float>(p.type);
        packed[offset + 9]  = static_cast<float>(p.material);
        packed[offset + 10] = p.ior;
        packed[offset + 11] = 0.0f;
        offset += kFloatsPerPrim;
    }
    device_->WriteBuffer(pt::rhi::BufferHandle{prim_buffer_id_},
                         packed.data(), packed.size() * sizeof(float));

    primitives_dirty_ = false;
    accum_dirty_      = true;
}

void Engine::RenderFrame() {
    if (!device_ || pathtrace_pipeline_id_ == 0) return;

    EnsureMeshUpdated();
    EnsurePrimitivesUploaded();

    auto& C = pt::console::Console::Get();

    // P10 denoiser state. Resolved before BeginFrame so we know whether
    // to allocate the G-buffer textures this frame.
    bool want_denoiser = false;
    if (auto* v = C.FindCVar("r_denoiser")) {
        want_denoiser = (v->value == "metalfx") && device_->SupportsDenoise();
    }
    if (want_denoiser != denoiser_active_) {
        // Toggle: free or allocate G-buffer next, force history reset.
        denoiser_active_      = want_denoiser;
        prev_view_proj_valid_ = false;
        if (!want_denoiser && device_) {
            if (denoise_color_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
            if (depth_tex_id_         != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
            if (motion_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
            denoise_color_tex_id_ = depth_tex_id_ = motion_tex_id_ = 0;
        }
    }

    auto fc = device_->BeginFrame();

    // Camera-movement detection -> reset accumulation.
    auto& cam = *camera_;
    bool cam_moved = (cam.pos.x != last_cam_pos_[0]) ||
                     (cam.pos.y != last_cam_pos_[1]) ||
                     (cam.pos.z != last_cam_pos_[2]) ||
                     (cam.yaw   != last_cam_yaw_)    ||
                     (cam.pitch != last_cam_pitch_)  ||
                     (cam.fov_deg != last_cam_fov_);
    if (cam_moved) accum_dirty_ = true;
    last_cam_pos_[0] = cam.pos.x; last_cam_pos_[1] = cam.pos.y; last_cam_pos_[2] = cam.pos.z;
    last_cam_yaw_   = cam.yaw;   last_cam_pitch_  = cam.pitch;  last_cam_fov_  = cam.fov_deg;

    // Lazily (re)create the HDR accumulation texture; reallocate if the
    // swapchain size changed.
    bool size_changed = (accum_w_ != static_cast<int>(fc.width)) ||
                        (accum_h_ != static_cast<int>(fc.height));
    if (accum_texture_id_ == 0 || size_changed) {
        if (accum_texture_id_ != 0) {
            device_->DestroyTexture(pt::rhi::TextureHandle{accum_texture_id_});
        }
        pt::rhi::TextureDesc td{
            .width  = fc.width,
            .height = fc.height,
            .format = pt::rhi::TextureFormat::RGBA32F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "accum_hdr",
        };
        auto h = device_->CreateTexture(td);
        accum_texture_id_ = h.id;
        if (accum_texture_id_ == 0) {
            LOG_ERROR("CreateTexture(RGBA32F {}x{}) failed", fc.width, fc.height);
            return;
        }
        accum_w_ = static_cast<int>(fc.width);
        accum_h_ = static_cast<int>(fc.height);
        accum_dirty_ = true;
    }

    // P10: G-buffer textures for the denoiser. Allocate (and re-allocate
    // on resize) only while r_denoiser is on -- saves ~32MB at 1080p when
    // denoising is off.
    if (denoiser_active_ &&
        (denoise_color_tex_id_ == 0 || depth_tex_id_ == 0 || motion_tex_id_ == 0 || size_changed)) {
        if (denoise_color_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        if (depth_tex_id_         != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
        if (motion_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
        auto color_h = device_->CreateTexture({
            .width = fc.width, .height = fc.height,
            .format = pt::rhi::TextureFormat::RGBA16F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "denoise_color",
        });
        auto depth_h = device_->CreateTexture({
            .width = fc.width, .height = fc.height,
            .format = pt::rhi::TextureFormat::R32F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "denoise_depth",
        });
        auto motion_h = device_->CreateTexture({
            .width = fc.width, .height = fc.height,
            .format = pt::rhi::TextureFormat::RG16F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "denoise_motion",
        });
        denoise_color_tex_id_ = color_h.id;
        depth_tex_id_         = depth_h.id;
        motion_tex_id_        = motion_h.id;
        prev_view_proj_valid_ = false;        // history is invalid after resize
        if (denoise_color_tex_id_ == 0 || depth_tex_id_ == 0 || motion_tex_id_ == 0) {
            LOG_ERROR("denoiser G-buffer allocation failed at {}x{}", fc.width, fc.height);
            denoiser_active_ = false;
        }
    }

    auto* cb = device_->AcquireCommandBuffer();
    cb->BindComputePipeline(pt::rhi::PipelineHandle{pathtrace_pipeline_id_});
    cb->BindStorageTexture(0, fc.swapchain_image);
    cb->BindStorageTexture(1, pt::rhi::TextureHandle{accum_texture_id_});

    // Slot mapping (matches PathTrace.slang and the Metal-buffer layout
    // Slang produces): AS at buffer(0), then storage buffers in declared
    // order. Our RHI uses separate "slot namespaces" so the user-facing
    // numbers below are 1-based for buffers (the AS, declared at vk slot
    // 2, lands at buffer(0) on Metal; mesh_positions at buffer(1), etc.).
    bool tlas_present = (scene_tlas_id_ != 0);
    if (tlas_present) {
        cb->BindAccelStruct(2, pt::rhi::AccelStructHandle{scene_tlas_id_});
        if (box_vbuf_id_ != 0) cb->BindBuffer(1, pt::rhi::BufferHandle{box_vbuf_id_}, 0);
        if (box_ibuf_id_ != 0) cb->BindBuffer(2, pt::rhi::BufferHandle{box_ibuf_id_}, 0);
    }
    if (prim_buffer_id_ != 0) {
        cb->BindBuffer(3, pt::rhi::BufferHandle{prim_buffer_id_}, 0);
    }
    // P10 G-buffer texture binds. The shader's vk::binding numbers (6/7/8)
    // are Vulkan descriptor slots; on Metal Slang assigns texture(N) in
    // declaration order, so output/accum/denoise_color/depth/motion/env
    // become texture(0..5). The Metal RHI treats the slot arg as the MSL
    // texture index, so we bind them at 2/3/4/5 here.
    if (denoiser_active_) {
        cb->BindStorageTexture(2, pt::rhi::TextureHandle{denoise_color_tex_id_});
        cb->BindStorageTexture(3, pt::rhi::TextureHandle{depth_tex_id_});
        cb->BindStorageTexture(4, pt::rhi::TextureHandle{motion_tex_id_});
    }
    if (env_map_tex_id_ != 0) {
        cb->BindStorageTexture(5, pt::rhi::TextureHandle{env_map_tex_id_});
    }

    std::uint32_t bounces = 8;
    if (auto* v = C.FindCVar("r_max_bounces")) bounces = (std::uint32_t)v->GetInt();
    std::uint32_t spp = 1;
    if (auto* v = C.FindCVar("r_spp")) {
        int n = v->GetInt();
        if (n < 1)  n = 1;
        if (n > 32) n = 32;     // clamp to keep a runaway value from freezing the GPU
        spp = static_cast<std::uint32_t>(n);
    }

    const auto fwd   = cam.Forward();
    const auto right = cam.Right();
    const auto up    = cam.Up();
    const float aspect = (fc.height > 0) ? float(fc.width) / float(fc.height) : 1.0f;

    // Build current view*projection. Used by the shader's G-buffer pass
    // to reproject hit world positions into both current and previous
    // screen space (motion vectors). Identity prev_view_proj on the
    // first frame yields zero motion, which the denoiser tolerates.
    glm::mat4 view = glm::lookAtRH(cam.pos, cam.pos + fwd, glm::vec3(0,1,0));
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(cam.fov_deg), aspect, 0.05f, 500.0f);
    glm::mat4 curr_view_proj = proj * view;

    struct PtPush {
        float pos_fovtan[4];
        float fwd_aspect[4];
        float right_xyz[4];
        float up_xyz[4];
        std::uint32_t frame_index;
        std::uint32_t reset_accum;
        std::uint32_t max_bounces;
        std::uint32_t tlas_present;
        std::uint32_t prim_count;
        std::uint32_t spp;
        std::uint32_t denoiser_enabled;
        std::uint32_t env_map_present;
        float halton_jitter[2];
        float env_intensity;
        float pad3;
        float curr_view_proj[16];
        float prev_view_proj[16];
    } push{};
    push.pos_fovtan[0] = cam.pos.x; push.pos_fovtan[1] = cam.pos.y;
    push.pos_fovtan[2] = cam.pos.z; push.pos_fovtan[3] = cam.FovYTan();
    push.fwd_aspect[0] = fwd.x; push.fwd_aspect[1] = fwd.y;
    push.fwd_aspect[2] = fwd.z; push.fwd_aspect[3] = aspect;
    push.right_xyz[0]  = right.x; push.right_xyz[1] = right.y;
    push.right_xyz[2]  = right.z; push.right_xyz[3] = 0.0f;
    push.up_xyz[0]     = up.x; push.up_xyz[1] = up.y;
    push.up_xyz[2]     = up.z; push.up_xyz[3] = 0.0f;
    push.frame_index   = frame_index_++;
    // When the denoiser is on it does its own temporal reuse, so we feed
    // it FRESH (un-accumulated) per-frame radiance every dispatch. The
    // shader still writes accum_hdr (so toggling denoiser off picks back
    // up the accumulating image) but reads it as if reset.
    push.reset_accum   = (denoiser_active_ || accum_dirty_) ? 1u : 0u;
    push.max_bounces   = bounces;
    push.tlas_present  = tlas_present ? 1u : 0u;
    push.prim_count    = static_cast<std::uint32_t>(primitives_.size());
    push.spp           = spp;
    push.denoiser_enabled = denoiser_active_ ? 1u : 0u;
    push.env_map_present  = (env_map_tex_id_ != 0) ? 1u : 0u;
    {
        float intensity = 1.0f;
        if (auto* v = C.FindCVar("r_env_intensity")) intensity = v->GetFloat();
        push.env_intensity = intensity;
    }
    push.pad3 = 0.0f;

    // Halton(2,3) sub-pixel jitter sequence in [-0.5, 0.5] each axis.
    // 16-sample period before repeating; ample for the denoiser's
    // internal history depth. Use frame_index_ which already advanced
    // above, so each frame's color ray is a unique sub-pixel sample.
    auto halton = [](std::uint32_t i, std::uint32_t base) -> float {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= float(base); r += f * float(i % base); i /= base; }
        return r;
    };
    std::uint32_t hi = (push.frame_index % 16u) + 1u;
    push.halton_jitter[0] = halton(hi, 2) - 0.5f;
    push.halton_jitter[1] = halton(hi, 3) - 0.5f;
    last_jitter_x_ = push.halton_jitter[0];
    last_jitter_y_ = push.halton_jitter[1];

    // pad3 already 0.0f from value-init
    const glm::mat4 prev_vp = prev_view_proj_valid_ ? prev_view_proj_ : curr_view_proj;
    std::memcpy(push.curr_view_proj, glm::value_ptr(curr_view_proj), sizeof(push.curr_view_proj));
    std::memcpy(push.prev_view_proj, glm::value_ptr(prev_vp),        sizeof(push.prev_view_proj));
    static_assert(sizeof(PtPush) == 240);
    cb->PushConstants(&push, sizeof(push));
    accum_dirty_ = false;

    auto wg_x = (fc.width  + 7) / 8;
    auto wg_y = (fc.height + 7) / 8;
    cb->Dispatch(wg_x, wg_y, 1);

    // P10 denoise pass: encoded onto the same command buffer between
    // the path tracer dispatch and Submit. MetalFX reads the G-buffer
    // textures the shader just wrote and outputs to the swapchain
    // (overwriting the shader's tonemapped fallback).
    if (denoiser_active_) {
        pt::rhi::Device::DenoiseDesc dd{};
        dd.color_in      = pt::rhi::TextureHandle{denoise_color_tex_id_};
        dd.depth_in      = pt::rhi::TextureHandle{depth_tex_id_};
        dd.motion_in     = pt::rhi::TextureHandle{motion_tex_id_};
        dd.output        = fc.swapchain_image;
        dd.jitter_x      = last_jitter_x_;
        dd.jitter_y      = last_jitter_y_;
        dd.reset_history = !prev_view_proj_valid_;
        // MetalFX wants worldToView and viewToClip separately, so pass
        // them rather than the combined view*proj. glm matrices are
        // column-major, matching simd_float4x4 on the consumer end.
        dd.world_to_view = glm::value_ptr(view);
        dd.view_to_clip  = glm::value_ptr(proj);
        device_->Denoise(dd);
    }

    device_->Submit(cb);
    device_->EndFrame(cb);

    prev_view_proj_       = curr_view_proj;
    prev_view_proj_valid_ = true;
}

void Engine::UpdateCamera(double dt) {
    if (!camera_ || !window_) return;
    auto& C = pt::console::Console::Get();

    // Mouse-look: hold right mouse to capture, release to free the cursor.
    bool rmb = window_->IsMouseButtonDown(1);   // GLFW_MOUSE_BUTTON_RIGHT == 1
    if (rmb && !mouse_look_active_) {
        window_->SetCursorMode(0x00034003);     // GLFW_CURSOR_DISABLED
        mouse_look_active_ = true;
    } else if (!rmb && mouse_look_active_) {
        window_->SetCursorMode(0x00034001);     // GLFW_CURSOR_NORMAL
        mouse_look_active_ = false;
    }

    if (mouse_look_active_) {
        double dx = 0.0, dy = 0.0;
        window_->ConsumeMouseDelta(dx, dy);
        float sens = 0.12f;
        if (auto* v = C.FindCVar("cam_sensitivity")) sens = v->GetFloat();
        camera_->yaw   += glm::radians(static_cast<float>(dx) * sens);
        camera_->pitch -= glm::radians(static_cast<float>(dy) * sens);
        camera_->ClampPitch();
    }

    // WASD + Space/Ctrl movement.  Speed in units/sec; shift sprints.
    float speed   = 3.0f;
    float sprint  = 3.0f;
    if (auto* v = C.FindCVar("cam_speed"))       speed  = v->GetFloat();
    if (auto* v = C.FindCVar("cam_sprint_mult")) sprint = v->GetFloat();

    bool shift = window_->IsKeyDown(340) || window_->IsKeyDown(344);  // L/R Shift
    if (shift) speed *= sprint;

    glm::vec3 fwd   = camera_->Forward();
    glm::vec3 right = camera_->Right();
    glm::vec3 wm{0.0f};
    if (window_->IsKeyDown(87))  wm += fwd;            // W
    if (window_->IsKeyDown(83))  wm -= fwd;            // S
    if (window_->IsKeyDown(68))  wm += right;          // D
    if (window_->IsKeyDown(65))  wm -= right;          // A
    if (window_->IsKeyDown(32))  wm += glm::vec3(0,1,0);  // Space
    if (window_->IsKeyDown(341)) wm -= glm::vec3(0,1,0);  // L Ctrl
    if (glm::dot(wm, wm) > 0.0f) {
        camera_->pos += glm::normalize(wm) * (speed * static_cast<float>(dt));
    }

    // Pull live overrides from cvars if the user typed them.  The
    // simplest UX is "if cvar string differs from camera state, apply".
    // We push back the camera state to the cvars on every frame too so
    // the panel reflects current values.
    if (auto* v = C.FindCVar("cam_fov")) {
        float new_fov = v->GetFloat();
        if (std::abs(new_fov - camera_->fov_deg) > 1e-3f) camera_->fov_deg = new_fov;
    }
}

void Engine::Tick(double dt) {
    pt::console::Console::Get().Drain();
    UpdateCamera(dt);

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

    C.RegisterCommand("scene_save",
        "scene_save <path.toml>: write camera + analytic primitives to a TOML file. (CSG state isn't saved yet -- put csg_* commands in autoexec.cfg if you want it to persist.)",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: scene_save <path.toml>"); return; }
            std::string path(args[0]);

            toml::table root;

            // Camera.
            auto& cam = *camera_;
            toml::array pos_arr{cam.pos.x, cam.pos.y, cam.pos.z};
            toml::table cam_tbl;
            cam_tbl.insert("pos",   std::move(pos_arr));
            cam_tbl.insert("yaw",   double(glm::degrees(cam.yaw)));
            cam_tbl.insert("pitch", double(glm::degrees(cam.pitch)));
            cam_tbl.insert("fov",   double(cam.fov_deg));
            root.insert("camera", std::move(cam_tbl));

            // Primitives.
            auto mat_name = [](AnalyticPrim::Material m) {
                switch (m) {
                    case AnalyticPrim::Lambert:    return "lambert";
                    case AnalyticPrim::Metal:      return "metal";
                    case AnalyticPrim::Dielectric: return "dielectric";
                }
                return "?";
            };
            toml::array prims_arr;
            for (const auto& [id, p] : primitives_) {
                toml::table t;
                t.insert("id", int64_t(id));
                t.insert("type", p.type == AnalyticPrim::Sphere ? "sphere" : "plane");
                toml::array pn{double(p.pos_or_n[0]), double(p.pos_or_n[1]), double(p.pos_or_n[2])};
                if (p.type == AnalyticPrim::Sphere) {
                    t.insert("center", std::move(pn));
                    t.insert("radius", double(p.radius_or_d));
                } else {
                    t.insert("normal", std::move(pn));
                    t.insert("d",      double(p.radius_or_d));
                }
                t.insert("material", mat_name(p.material));
                t.insert("color", toml::array{double(p.albedo[0]), double(p.albedo[1]), double(p.albedo[2])});
                t.insert("roughness", double(p.roughness));
                t.insert("ior",       double(p.ior));
                prims_arr.push_back(std::move(t));
            }
            root.insert("primitives", std::move(prims_arr));

            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f.is_open()) { out.FormatLine("scene_save: cannot open '{}'", path); return; }
            f << "# DeMonT Engine scene -- generated by scene_save\n\n";
            f << root;
            out.FormatLine("scene: saved {} primitive(s) + camera to {}", primitives_.size(), path);
        });

    C.RegisterCommand("scene_load",
        "scene_load <path.toml>: replace current camera + analytic primitives with the contents of a TOML scene file.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: scene_load <path.toml>"); return; }
            std::string path(args[0]);

            auto parsed = toml::parse_file(path);
            if (!parsed) {
                out.FormatLine("scene_load: parse error in {}: {}",
                               path, std::string(parsed.error().description()));
                return;
            }
            const toml::table& root = parsed.table();

            // Camera (optional).
            if (auto* cam_node = root.get("camera"); cam_node && cam_node->is_table()) {
                const auto& ct = *cam_node->as_table();
                if (auto pos = ct["pos"].as_array(); pos && pos->size() == 3) {
                    camera_->pos = glm::vec3(
                        float((*pos)[0].value_or(0.0)),
                        float((*pos)[1].value_or(0.0)),
                        float((*pos)[2].value_or(0.0)));
                }
                if (auto v = ct["yaw"].value<double>())   camera_->yaw     = glm::radians(float(*v));
                if (auto v = ct["pitch"].value<double>()) camera_->pitch   = glm::radians(float(*v));
                if (auto v = ct["fov"].value<double>())   camera_->fov_deg = float(*v);
                camera_->ClampPitch();
            }

            // Primitives (replaces the entire set).
            primitives_.clear();
            if (auto* prims_node = root.get("primitives"); prims_node && prims_node->is_array()) {
                for (const auto& el : *prims_node->as_array()) {
                    if (!el.is_table()) continue;
                    const auto& t = *el.as_table();
                    AnalyticPrim p{};
                    p.type = (t["type"].value_or<std::string>("sphere") == "plane")
                                 ? AnalyticPrim::Plane : AnalyticPrim::Sphere;
                    auto pn_key = (p.type == AnalyticPrim::Plane) ? "normal" : "center";
                    if (auto pn = t[pn_key].as_array(); pn && pn->size() == 3) {
                        p.pos_or_n[0] = float((*pn)[0].value_or(0.0));
                        p.pos_or_n[1] = float((*pn)[1].value_or(0.0));
                        p.pos_or_n[2] = float((*pn)[2].value_or(0.0));
                    }
                    auto r_key = (p.type == AnalyticPrim::Plane) ? "d" : "radius";
                    p.radius_or_d = float(t[r_key].value_or<double>(0.5));
                    auto mat = t["material"].value_or<std::string>("lambert");
                    if      (mat == "metal")      p.material = AnalyticPrim::Metal;
                    else if (mat == "dielectric") p.material = AnalyticPrim::Dielectric;
                    else                          p.material = AnalyticPrim::Lambert;
                    if (auto c = t["color"].as_array(); c && c->size() == 3) {
                        p.albedo[0] = float((*c)[0].value_or(1.0));
                        p.albedo[1] = float((*c)[1].value_or(1.0));
                        p.albedo[2] = float((*c)[2].value_or(1.0));
                    }
                    p.roughness = float(t["roughness"].value_or<double>(0.0));
                    p.ior       = float(t["ior"].value_or<double>(1.5));
                    auto id = std::uint32_t(t["id"].value_or<int64_t>(int64_t(primitives_.size() + 1)));
                    primitives_[id] = p;
                }
            }
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("scene: loaded {} primitive(s) from {}", primitives_.size(), path);
        });

    C.RegisterCommand("screenshot",
        "screenshot <path.ppm> [accum|denoise_color|depth|motion]: dump the target render texture to a PPM file (P6 binary, ACES-tonemapped for HDR inputs).",
        [this](auto args, pt::console::Output& out) {
            if (!device_) { out.PrintLine("screenshot: no device"); return; }
            if (args.empty()) { out.PrintLine("usage: screenshot <path.ppm> [accum|denoise_color|depth|motion]"); return; }
            std::string path(args[0]);
            std::string target = (args.size() >= 2) ? std::string(args[1]) : std::string("accum");

            std::uint64_t tex_id = 0;
            int channels = 4;
            int bytes_per_pixel = 0;
            const char* tag = nullptr;
            if (target == "accum") {
                tex_id = accum_texture_id_;
                bytes_per_pixel = 16;   // RGBA32F
                tag = "accum_hdr";
            } else if (target == "denoise_color") {
                if (!denoiser_active_) { out.PrintLine("screenshot: denoiser is off; enable r_denoiser metalfx first"); return; }
                tex_id = denoise_color_tex_id_;
                bytes_per_pixel = 8;    // RGBA16F
                tag = "denoise_color";
            } else if (target == "depth") {
                if (!denoiser_active_) { out.PrintLine("screenshot: denoiser is off"); return; }
                tex_id = depth_tex_id_;
                bytes_per_pixel = 4;    // R32F
                channels = 1;
                tag = "depth";
            } else if (target == "motion") {
                if (!denoiser_active_) { out.PrintLine("screenshot: denoiser is off"); return; }
                tex_id = motion_tex_id_;
                bytes_per_pixel = 4;    // RG16F
                channels = 2;
                tag = "motion";
            } else {
                out.FormatLine("screenshot: unknown target '{}'", target);
                return;
            }
            if (tex_id == 0) {
                out.FormatLine("screenshot: target '{}' not allocated yet", target);
                return;
            }

            std::uint32_t w = 0, hgt = 0;
            std::vector<std::uint8_t> raw(std::size_t(accum_w_) * accum_h_ * bytes_per_pixel);
            bool ok = device_->ReadbackTexture(pt::rhi::TextureHandle{tex_id},
                                               raw.data(), raw.size(), &w, &hgt);
            if (!ok || w == 0 || hgt == 0) {
                out.PrintLine("screenshot: ReadbackTexture failed");
                return;
            }

            // Convert to 8-bit RGB. ACES tonemap for HDR; depth gets a
            // grayscale ramp (0=black, 1=white); motion encodes (R = +x
            // mapped to [0..1], G = +y mapped, B=0) so directionality
            // is visible. Half-float decode is via __fp16 (Apple Clang).
            auto tonemap = [](float c) -> std::uint8_t {
                const float a = 2.51f, b = 0.03f, d = 2.43f, e = 0.59f, f = 0.14f;
                float x = (c * (a * c + b)) / (c * (d * c + e) + f);
                if (x < 0.0f) x = 0.0f;
                if (x > 1.0f) x = 1.0f;
                return static_cast<std::uint8_t>(x * 255.0f + 0.5f);
            };
            auto half_to_float = [](std::uint16_t h_) -> float {
                __fp16 v;
                std::memcpy(&v, &h_, 2);
                return float(v);
            };

            std::vector<std::uint8_t> rgb(std::size_t(w) * hgt * 3);
            for (std::uint32_t y = 0; y < hgt; ++y) {
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::size_t pi = std::size_t(y) * w + x;
                    float r = 0, g = 0, b = 0;
                    if (target == "accum") {
                        const float* src = reinterpret_cast<const float*>(raw.data()) + pi * 4;
                        r = src[0]; g = src[1]; b = src[2];
                    } else if (target == "denoise_color") {
                        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(raw.data()) + pi * 4;
                        r = half_to_float(src[0]);
                        g = half_to_float(src[1]);
                        b = half_to_float(src[2]);
                    } else if (target == "depth") {
                        const float* src = reinterpret_cast<const float*>(raw.data()) + pi;
                        float d_v = src[0];
                        if (d_v > 1.0f) d_v = 1.0f;   // clamp far points so depth>1 still visible
                        r = g = b = d_v;
                    } else { // motion
                        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(raw.data()) + pi * 2;
                        float mx = half_to_float(src[0]);
                        float my = half_to_float(src[1]);
                        // Map a +/-32 pixel range into [0,1] each so a
                        // grey image means "no motion".
                        r = 0.5f + mx / 64.0f;
                        g = 0.5f + my / 64.0f;
                        b = 0.0f;
                    }
                    std::uint8_t* dst = rgb.data() + pi * 3;
                    if (target == "depth" || target == "motion") {
                        // No tonemap; values already in [0,1] range.
                        auto clamp01 = [](float v) -> std::uint8_t {
                            if (v < 0) v = 0;
                            if (v > 1) v = 1;
                            return std::uint8_t(v * 255.0f + 0.5f);
                        };
                        dst[0] = clamp01(r); dst[1] = clamp01(g); dst[2] = clamp01(b);
                    } else {
                        dst[0] = tonemap(r); dst[1] = tonemap(g); dst[2] = tonemap(b);
                    }
                }
            }

            FILE* f = std::fopen(path.c_str(), "wb");
            if (f == nullptr) { out.FormatLine("screenshot: cannot open '{}'", path); return; }
            std::fprintf(f, "P6\n%u %u\n255\n", w, hgt);
            std::fwrite(rgb.data(), 1, rgb.size(), f);
            std::fclose(f);
            out.FormatLine("screenshot: wrote {} ({}x{} {})", path, w, hgt, tag);
            (void)channels;
        });

    C.RegisterCommand("web_console",
        "Open the web console in the default browser.",
        [this](auto, pt::console::Output& out) {
            OpenWebConsole();
            out.PrintLine("opening web console...");
        });

    C.RegisterCommand("toggle",
        "toggle <cvar>: cycle a cvar through its allowed_values (great for A/B testing).",
        [](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: toggle <cvar>"); return; }
            auto& C2 = pt::console::Console::Get();
            auto* v = C2.FindCVar(args[0]);
            if (v == nullptr) { out.FormatLine("toggle: unknown cvar '{}'", args[0]); return; }
            if (v->allowed_values.empty()) {
                out.FormatLine("toggle: '{}' has no allowed_values; can't cycle", v->name);
                return;
            }
            // Find the current value's index in allowed_values; advance one
            // (with wrap). If the current value isn't in the list (rare),
            // pick the first allowed value.
            std::size_t cur = v->allowed_values.size();   // sentinel
            for (std::size_t i = 0; i < v->allowed_values.size(); ++i) {
                if (v->allowed_values[i] == v->value) { cur = i; break; }
            }
            std::size_t next = (cur >= v->allowed_values.size()) ? 0
                                                                  : (cur + 1) % v->allowed_values.size();
            const std::string& chosen = v->allowed_values[next];
            std::string before = v->value;
            C2.SetCVarOverride(v->name, chosen);
            out.FormatLine("{} : \"{}\" -> \"{}\"", v->name, before, chosen);
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
    // r_theme: validate + push the new theme to every WS client so the
    // browser console flips live without a page reload.
    if (auto* v = C.FindCVar("r_theme")) {
        v->allowed_values = {"hardcore", "amber", "synthwave",
                             "matrix", "vault", "sakura", "mono"};
        v->on_change = [this](const pt::console::CVar& cv) {
            if (server_) {
                auto data = fmt::format(R"({{"name":"{}"}})", cv.value);
                server_->BroadcastEvent("theme_change", data);
            }
            if (overlay_) overlay_->ApplyTheme(cv.value);
        };
    }
    // dev_log_level: validate
    if (auto* v = C.FindCVar("dev_log_level")) {
        v->allowed_values = {"error", "warn", "info", "debug"};
    }
    if (auto* v = C.FindCVar("r_denoiser")) {
        v->allowed_values = {"off", "metalfx"};
    }
    // r_env_map: hot-reload on change. Empty -> unload (procedural sky).
    if (auto* v = C.FindCVar("r_env_map")) {
        v->on_change = [this](const pt::console::CVar& cv) {
            ReloadEnvMap(cv.value);
        };
    }
    if (auto* v = C.FindCVar("r_env_intensity")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // app_vsync / app_overlay_enabled / app_auto_open_console / dev_cheats:
    // boolean toggles -- accept 0|1.
    for (const char* n : {"app_vsync", "app_overlay_enabled",
                          "app_auto_open_console", "dev_cheats"}) {
        if (auto* v = C.FindCVar(n)) v->allowed_values = {"0", "1"};
    }

    RegisterCsgCommands();
    RegisterPrimCommands();
}

namespace {

// Strip-leading parser for a positive integer node id.
bool ParseUint(std::string_view s, std::uint32_t& out) {
    if (s.empty()) return false;
    std::uint32_t v = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size()) return false;
    out = v;
    return true;
}

bool ParseFloat(std::string_view s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::string buf(s);
    float v = std::strtof(buf.c_str(), &end);
    if (end == buf.c_str()) return false;
    out = v;
    return true;
}

}  // namespace

void Engine::RegisterCsgCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("csg_dump",
        "Print the current CSG tree.",
        [this](auto, pt::console::Output& out) {
            if (!csg_scene_) { out.PrintLine("CSG scene not initialised"); return; }
            std::string body;
            csg_scene_->Dump(body);
            out.Print(body);
        });

    C.RegisterCommand("csg_reset",
        "Reset the CSG scene to the default drilled-cube.",
        [this](auto, pt::console::Output& out) {
            if (!csg_scene_) { out.PrintLine("CSG scene not initialised"); return; }
            SeedDefaultCsgScene();
            out.PrintLine("csg: reset to drilled cube (root = 3, box - sphere)");
        });

    C.RegisterCommand("csg_box",
        "csg_box <id> <sx> <sy> <sz> <tx> <ty> <tz>: add a box leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 7) { out.PrintLine("usage: csg_box <id> <sx> <sy> <sz> <tx> <ty> <tz>"); return; }
            std::uint32_t id;
            float sx, sy, sz, tx, ty, tz;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], sx) || !ParseFloat(args[2], sy) || !ParseFloat(args[3], sz) ||
                !ParseFloat(args[4], tx) || !ParseFloat(args[5], ty) || !ParseFloat(args[6], tz)) {
                out.PrintLine("csg_box: arg parse failed");
                return;
            }
            if (!csg_scene_->AddBox(id, sx, sy, sz, tx, ty, tz)) {
                out.FormatLine("csg_box: id {} already exists or extents are non-positive", id);
                return;
            }
            out.FormatLine("csg: added box id={} ({}x{}x{} @ {} {} {})", id, sx, sy, sz, tx, ty, tz);
        });

    C.RegisterCommand("csg_sphere",
        "csg_sphere <id> <radius> <segments> <tx> <ty> <tz>: add a sphere leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 6) { out.PrintLine("usage: csg_sphere <id> <radius> <segments> <tx> <ty> <tz>"); return; }
            std::uint32_t id, segs;
            float radius, tx, ty, tz;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], radius) ||
                !ParseUint(args[2], segs) ||
                !ParseFloat(args[3], tx) || !ParseFloat(args[4], ty) || !ParseFloat(args[5], tz)) {
                out.PrintLine("csg_sphere: arg parse failed");
                return;
            }
            if (!csg_scene_->AddSphere(id, radius, static_cast<int>(segs), tx, ty, tz)) {
                out.FormatLine("csg_sphere: id {} already exists or radius is non-positive", id);
                return;
            }
            out.FormatLine("csg: added sphere id={} (r={} segs={} @ {} {} {})", id, radius, segs, tx, ty, tz);
        });

    C.RegisterCommand("csg_cylinder",
        "csg_cylinder <id> <radius> <height> <segments> <tx> <ty> <tz>: add a cylinder leaf.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 7) { out.PrintLine("usage: csg_cylinder <id> <radius> <height> <segments> <tx> <ty> <tz>"); return; }
            std::uint32_t id, segs;
            float radius, height, tx, ty, tz;
            if (!ParseUint(args[0], id) || !ParseFloat(args[1], radius) || !ParseFloat(args[2], height) ||
                !ParseUint(args[3], segs) ||
                !ParseFloat(args[4], tx) || !ParseFloat(args[5], ty) || !ParseFloat(args[6], tz)) {
                out.PrintLine("csg_cylinder: arg parse failed");
                return;
            }
            if (!csg_scene_->AddCylinder(id, radius, height, static_cast<int>(segs), tx, ty, tz)) {
                out.FormatLine("csg_cylinder: id {} already exists or extents are non-positive", id);
                return;
            }
            out.FormatLine("csg: added cylinder id={} (r={} h={} segs={} @ {} {} {})",
                           id, radius, height, segs, tx, ty, tz);
        });

    C.RegisterCommand("csg_op",
        "csg_op <id> <union|subtract|intersect> <left_id> <right_id>: combine two nodes.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 4) { out.PrintLine("usage: csg_op <id> <union|subtract|intersect> <left> <right>"); return; }
            std::uint32_t id, l, r;
            if (!ParseUint(args[0], id) || !ParseUint(args[2], l) || !ParseUint(args[3], r)) {
                out.PrintLine("csg_op: id parse failed");
                return;
            }
            pt::csg::OpType op;
            if      (args[1] == "union")     op = pt::csg::OpType::Union;
            else if (args[1] == "subtract")  op = pt::csg::OpType::Subtract;
            else if (args[1] == "intersect") op = pt::csg::OpType::Intersect;
            else { out.FormatLine("csg_op: unknown op '{}' (want union|subtract|intersect)", args[1]); return; }
            if (!csg_scene_->Combine(id, op, l, r)) {
                out.FormatLine("csg_op: id {} already exists or operands {} / {} missing", id, l, r);
                return;
            }
            out.FormatLine("csg: combined id={} = {}({}, {})", id, pt::csg::OpName(op), l, r);
        });

    C.RegisterCommand("csg_remove",
        "csg_remove <id>: drop a node (and any internal nodes referencing it).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: csg_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("csg_remove: id parse failed"); return; }
            std::size_t n = csg_scene_->Remove(id);
            if (n == 0) out.FormatLine("csg_remove: id {} not found", id);
            else        out.FormatLine("csg: removed {} node(s) (cascaded from {})", n, id);
        });

    C.RegisterCommand("csg_set_root",
        "csg_set_root <id>: render the subtree rooted at <id>.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: csg_set_root <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("csg_set_root: id parse failed"); return; }
            if (!csg_scene_->SetRoot(id)) {
                out.FormatLine("csg_set_root: id {} not found", id);
                return;
            }
            out.FormatLine("csg: root set to {}", id);
        });
}

namespace {

bool ParseMaterial(std::string_view s, Engine::AnalyticPrim::Material& out) {
    if (s == "lambert")    { out = Engine::AnalyticPrim::Lambert;    return true; }
    if (s == "metal")      { out = Engine::AnalyticPrim::Metal;      return true; }
    if (s == "dielectric") { out = Engine::AnalyticPrim::Dielectric; return true; }
    return false;
}
const char* MaterialName(Engine::AnalyticPrim::Material m) {
    switch (m) {
        case Engine::AnalyticPrim::Lambert:    return "lambert";
        case Engine::AnalyticPrim::Metal:      return "metal";
        case Engine::AnalyticPrim::Dielectric: return "dielectric";
    }
    return "?";
}

}  // namespace

void Engine::RegisterPrimCommands() {
    auto& C = pt::console::Console::Get();

    C.RegisterCommand("prim_list",
        "List analytic primitives (sphere/plane).",
        [this](auto, pt::console::Output& out) {
            if (primitives_.empty()) { out.PrintLine("(no primitives)"); return; }
            out.FormatLine("{} primitive(s):", primitives_.size());
            for (const auto& [id, p] : primitives_) {
                if (p.type == AnalyticPrim::Sphere) {
                    out.FormatLine("  {:>3}  sphere  c=({:.2f} {:.2f} {:.2f}) r={:.3f}  {}  rgb=({:.2f} {:.2f} {:.2f}) rough={:.2f} ior={:.2f}",
                                   id,
                                   p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2],
                                   p.radius_or_d,
                                   MaterialName(p.material),
                                   p.albedo[0], p.albedo[1], p.albedo[2],
                                   p.roughness, p.ior);
                } else {
                    out.FormatLine("  {:>3}  plane   n=({:.2f} {:.2f} {:.2f}) d={:.3f}  {}  rgb=({:.2f} {:.2f} {:.2f})",
                                   id,
                                   p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2],
                                   p.radius_or_d,
                                   MaterialName(p.material),
                                   p.albedo[0], p.albedo[1], p.albedo[2]);
                }
            }
        });

    C.RegisterCommand("prim_clear",
        "Remove all analytic primitives.",
        [this](auto, pt::console::Output& out) {
            primitives_.clear();
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.PrintLine("primitives: cleared");
        });

    C.RegisterCommand("prim_reset",
        "Reset analytic primitives to the default 3-sphere + ground scene.",
        [this](auto, pt::console::Output& out) {
            SeedDefaultPrimitives();
            out.PrintLine("primitives: reset to default (red Lambert, gold metal, glass + ground)");
        });

    C.RegisterCommand("prim_remove",
        "prim_remove <id>: drop a primitive.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 1) { out.PrintLine("usage: prim_remove <id>"); return; }
            std::uint32_t id;
            if (!ParseUint(args[0], id)) { out.PrintLine("prim_remove: id parse failed"); return; }
            if (primitives_.erase(id) == 0) {
                out.FormatLine("prim_remove: id {} not found", id);
                return;
            }
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: removed id {}", id);
        });

    C.RegisterCommand("prim_sphere",
        "prim_sphere <id> <x> <y> <z> <radius> <lambert|metal|dielectric> <r> <g> <b> [roughness] [ior]: add or replace a sphere.",
        [this](auto args, pt::console::Output& out) {
            if (args.size() < 9 || args.size() > 11) {
                out.PrintLine("usage: prim_sphere <id> <x> <y> <z> <radius> <material> <r> <g> <b> [roughness=0] [ior=1.5]");
                return;
            }
            std::uint32_t id;
            float x, y, z, radius, r, g, b;
            float roughness = 0.0f, ior = 1.5f;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], x) || !ParseFloat(args[2], y) || !ParseFloat(args[3], z) ||
                !ParseFloat(args[4], radius) ||
                !ParseFloat(args[6], r) || !ParseFloat(args[7], g) || !ParseFloat(args[8], b)) {
                out.PrintLine("prim_sphere: arg parse failed");
                return;
            }
            AnalyticPrim::Material mat;
            if (!ParseMaterial(args[5], mat)) {
                out.FormatLine("prim_sphere: unknown material '{}' (want lambert|metal|dielectric)", args[5]);
                return;
            }
            if (radius <= 0.0f) { out.PrintLine("prim_sphere: radius must be > 0"); return; }
            if (args.size() >= 10 && !ParseFloat(args[9],  roughness)) { out.PrintLine("prim_sphere: bad roughness"); return; }
            if (args.size() >= 11 && !ParseFloat(args[10], ior))       { out.PrintLine("prim_sphere: bad ior");       return; }

            AnalyticPrim p{};
            p.type        = AnalyticPrim::Sphere;
            p.material    = mat;
            p.pos_or_n[0] = x; p.pos_or_n[1] = y; p.pos_or_n[2] = z;
            p.radius_or_d = radius;
            p.albedo[0]   = r; p.albedo[1] = g; p.albedo[2] = b;
            p.roughness   = roughness;
            p.ior         = ior;
            primitives_[id] = p;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: sphere id={} ({} @ {:.2f} {:.2f} {:.2f} r={})",
                           id, MaterialName(mat), x, y, z, radius);
        });

    C.RegisterCommand("prim_plane",
        "prim_plane <id> <nx> <ny> <nz> <d> <lambert|metal|dielectric> <r> <g> <b>: add or replace an infinite plane (n . p + d = 0).",
        [this](auto args, pt::console::Output& out) {
            if (args.size() != 9) {
                out.PrintLine("usage: prim_plane <id> <nx> <ny> <nz> <d> <material> <r> <g> <b>");
                return;
            }
            std::uint32_t id;
            float nx, ny, nz, d, r, g, b;
            if (!ParseUint(args[0], id) ||
                !ParseFloat(args[1], nx) || !ParseFloat(args[2], ny) || !ParseFloat(args[3], nz) ||
                !ParseFloat(args[4], d) ||
                !ParseFloat(args[6], r) || !ParseFloat(args[7], g) || !ParseFloat(args[8], b)) {
                out.PrintLine("prim_plane: arg parse failed");
                return;
            }
            AnalyticPrim::Material mat;
            if (!ParseMaterial(args[5], mat)) {
                out.FormatLine("prim_plane: unknown material '{}'", args[5]);
                return;
            }
            const float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len < 1e-6f) { out.PrintLine("prim_plane: normal magnitude is zero"); return; }
            nx /= len; ny /= len; nz /= len;

            AnalyticPrim p{};
            p.type        = AnalyticPrim::Plane;
            p.material    = mat;
            p.pos_or_n[0] = nx; p.pos_or_n[1] = ny; p.pos_or_n[2] = nz;
            p.radius_or_d = d;
            p.albedo[0]   = r; p.albedo[1] = g; p.albedo[2] = b;
            p.roughness   = 0.0f;
            p.ior         = 1.0f;
            primitives_[id] = p;
            primitives_dirty_ = true;
            accum_dirty_      = true;
            out.FormatLine("primitives: plane id={} ({} n=({:.2f} {:.2f} {:.2f}) d={:.3f})",
                           id, MaterialName(mat), nx, ny, nz, d);
        });
}

}  // namespace pt::engine

#include "Engine.h"

#include "../app/ConsoleOverlay.h"
#include "../app/Window.h"
#include "../console/Console.h"
#include "../console/ConsoleServer.h"
#include "../core/Hardware/HardwareInfo.h"
#include "../core/Jobs/JobSystem.h"
#include "../core/Log.h"
#include "../core/Memory/Memory.h"
#include "../renderer/Astronomy.h"
#include "../renderer/BscCatalog.h"
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
#include <ctime>
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
    PT_CVAR(r_quality,         "high",  "Master quality preset that drives r_spp, r_max_bounces, r_caustics, r_refract_bounces, etc. Options: low (fast, no caustics), medium (default-ish), high (caustics, more bounces), ultra (max). 'custom' leaves per-feature cvars as-is.", CVAR_ARCHIVE);
    PT_CVAR(r_caustics,        "1",  "Refractive shadow rays. 1 = NEE rays refract through dielectrics so glass/diamond produce caustic patterns; 0 = treat all dielectrics as opaque shadow blockers (faster, blocks any caustic). Path-tracer-correct in both modes.", CVAR_ARCHIVE);
    PT_CVAR(r_refract_bounces, "4",  "Maximum dielectric refractions a single shadow ray may chain through before giving up (returns no contribution). Higher catches more multi-facet caustics; lower is faster.", CVAR_ARCHIVE);
    PT_CVAR(r_denoiser,        "off","Denoiser: off|metalfx (Mac MetalFX TemporalDenoisedScaler).", CVAR_ARCHIVE);
    PT_CVAR(r_hdr_pipeline,    "1",  "Linear-HDR pipeline through MetalFX. 1 = path tracer writes raw HDR, MetalFX denoises in HDR, post-pass applies exposure+ACES (recommended). 0 = path tracer pre-applies exposure+ACES, MetalFX denoises LDR, tonemap pass is a passthrough copy. Only affects the denoiser-on path.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom,           "1",  "HDR bloom (downsample/upsample pyramid, additive composite before ACES). 0 disables; tonemap then samples a 1x1 zero buffer.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_threshold, "1.0","Linear-HDR luminance threshold for the bloom extract. Pixels below this value contribute nothing to the pyramid; pixels above contribute proportional to (lum - threshold). The path tracer's pixels are in tonemap-relative units (sun ~30, env ~3) so a threshold of 1.0 picks up only HDR highlights.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_intensity, "0.05","Linear blend factor of the bloom layer added on top of the HDR image before tonemap. 0 disables, 1 makes the bloom layer dominate. Realistic camera lens flare is in the 0.02-0.10 range.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_mips,      "5",  "How many mip levels the bloom pyramid uses (1..5). More mips = a softer / wider halo; fewer mips = a tighter glow. Capped to kBloomMips at compile time.", CVAR_ARCHIVE);
    PT_CVAR(r_bloom_radius,    "1.0","Per-mip upsample 'spread' multiplier. 1.0 = pixel-accurate dual-filter blur; >1 widens each upsample tap (softer, more diffuse halo); <1 tightens it (sharper core, less spread). Real range 0.5..3.0.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare,      "0",   "Lens flare. Image-based ghost reflections sampled from the bloom layer at mirror-across-centre positions. 0 disables.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_intensity, "0.15", "Linear blend strength of the flare layer. Real-camera lens flare is typically 0.1-0.3 of the bright source.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_dispersion, "0.012", "Per-channel scale offset for chromatic aberration on ghosts. 0 = achromatic (white ghosts), >0 = colourful rainbow fringe along ghost edges. Real lenses 0.01-0.03.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_count,"4",   "Number of ghost reflections to render (1..6). Each ghost has a different scale + colour tint hardcoded in the shader.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_threshold, "0.0", "(image mode only) Per-ghost luminance gate. 0 = no gate. 'sun' mode ignores this and draws clean sun-only flare.", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_mode, "sun", "Lens flare algorithm. 'sun' = explicit sun-position flare: projects the sun direction to screen and draws soft Gaussian-disc ghosts at its mirrored positions (clean, no scene mirroring, the production-correct approach). 'image' = old image-based flare: mirror-samples the full bloom mip at every output pixel (looks dreamy but reflects the whole scene as flare, which isn't physically what a lens does).", CVAR_ARCHIVE);
    PT_CVAR(r_lens_flare_size, "0.10", "(sun mode) Gaussian disc radius in normalised screen units. 0.05 = tight, 0.20 = soft and wide. Each ghost's radius scales with its distance from centre, so smaller scales naturally make tighter discs.", CVAR_ARCHIVE);
    PT_CVAR(r_exposure,        "1.5","Manual HDR exposure multiplier applied before ACES tonemap. Used when r_auto_exposure = 0.", CVAR_ARCHIVE);
    PT_CVAR(r_auto_exposure,   "1",  "Auto-exposure: 0 = use r_exposure manual value, 1 = sample accum_hdr each frame and adapt exposure toward r_exposure_target (eye-adaptation feel).", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_min,    "0.05",  "Minimum exposure scalar that auto-exposure can settle on. Stops a nuclear-bright scene from being crushed below this value.", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_max,    "4.0",   "Maximum exposure scalar that auto-exposure can settle on. The reason nights stay dark instead of being boosted to look like day -- bumping this lets the eye adapt further into the dark, lowering it caps the boost.", CVAR_ARCHIVE);
    PT_CVAR(r_exposure_target, "0.18",  "Middle-grey target for auto-exposure. 0.18 matches the Zone-V/Munsell middle-grey convention; lower values aim for a darker overall look.", CVAR_ARCHIVE);
    PT_CVAR(r_eye_adapt_speed, "0.20",  "Per-update interpolation factor for auto-exposure (0..1). Smaller = slower eye adaptation. The update fires every 8 frames, so 0.20 is roughly 'fully adapted in 1 second at 60fps'.", CVAR_ARCHIVE);
    PT_CVAR(r_eye_model,       "human", "Preset 'iris/lens' tuning: human (default), cat (better dim-light dynamic range), owl (nocturnal -- huge max), dslr_iso100 (locked, narrow), dslr_iso6400 (locked, high gain), phone (auto, modest range), linear (no tonemap, debug). Selecting a preset writes r_exposure_min/max/target/r_eye_adapt_speed; 'custom' leaves them as-is.", CVAR_ARCHIVE);
    PT_CVAR(r_env_map,         "assets/hdri/sunset.hdr",
            "Path to a Radiance .hdr environment map. Used when r_sky_mode = hdri. "
            "Default points at the bundled CC0 sunset HDRI; resolves relative to CWD.",
            CVAR_ARCHIVE);
    PT_CVAR(r_env_intensity,   "1.0","Scalar multiplier on env-map samples. Useful for darkening/brightening the IBL without re-authoring the HDRI.", CVAR_ARCHIVE);

    // Procedural sky (Preetham-lite analytic). Used when r_sky_mode is
    // "procedural". The sun position drives both the sky colour gradient
    // and the disk; positive elevation = above horizon (day), negative =
    // below (night); azimuth in degrees, 0 = -Z (north), 90 = +X (east).
    PT_CVAR(r_sky_mode,        "hdri", "Sky rendering: gradient (cheap fallback) | hdri (sample r_env_map) | procedural (analytic with sun position). Defaults to hdri so the bundled sunset.hdr is visible out of the box.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_elevation,   "30.0", "Sun elevation in degrees above horizon (-90..90). Drives day/sunset/night blend in procedural sky. Overridden when r_sky_use_astronomical = 1.", CVAR_ARCHIVE);
    PT_CVAR(r_sun_azimuth,     "135.0","Sun azimuth in degrees (0=north, 90=east, 180=south, 270=west). Overridden when r_sky_use_astronomical = 1.", CVAR_ARCHIVE);

    // Astronomical sky parameters. r_sky_use_astronomical=1 overrides
    // the manual sun cvars with positions computed from observer
    // lat/lon and current UTC time. Default location: Chennai, India.
    PT_CVAR(r_sky_use_astronomical, "0",
            "1 = compute sun position from r_sky_lat/r_sky_lon and current UTC, ignoring r_sun_elevation/r_sun_azimuth. 0 = use manual sun cvars.",
            CVAR_ARCHIVE);
    PT_CVAR(r_sky_lat,         "13.0827", "Observer latitude in degrees (+N). Default: Chennai, India.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_lon,         "80.2707", "Observer longitude in degrees (+E). Default: Chennai, India.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_hour,        "12.0",    "Hour of day (0..24) for the astronomical sun + starmap. Interpreted in UTC if r_sky_hour_local = 0, else in the selected city's local time. r_sky_animate = 1 advances this every frame.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_hour_local,  "1",       "If 1, r_sky_hour is the city's local time (using r_sky_tz_offset_hours, set by r_sky_city). If 0, UTC.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_tz_offset_hours, "5.5", "Selected city's UTC offset in hours (positive east). Updated when r_sky_city changes; you generally don't write this directly.", CVAR_READONLY);
    PT_CVAR(r_sky_animate,     "0",       "If 1, advance r_sky_hour every frame at r_sky_animate_rate hours per real-time second. Wraps at 24.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_animate_rate,"0.5",     "Hours of sim time per real-time second when r_sky_animate = 1. 0.5 = half-hour/s (a full day in 48s). 24 = compress a day into 1s. 1/3600 ≈ live-time.", CVAR_ARCHIVE);
    PT_CVAR(r_sky_city,        "chennai", "Preset observer location. Selecting one writes r_sky_lat / r_sky_lon to that city's coordinates. 'custom' leaves them as-is.", CVAR_ARCHIVE);
    PT_CVAR(r_show_stars,      "1",       "Render stars at night (sun below horizon). 0 disables.", CVAR_ARCHIVE);
    PT_CVAR(r_stars_mode,      "bsc",     "Star source: 'bsc' = real Yale Bright Star Catalog (J2000-frame map rotated to local horizon per frame, requires assets/stars/BSC5.dat), 'procedural' = hash-based random starfield (fast, no catalog needed). Falls back to procedural when 'bsc' is requested but the catalog failed to load.", CVAR_ARCHIVE);
    PT_CVAR(r_stars_twinkle,   "1",       "Per-star atmospheric scintillation. 0 = static field, 1 = each star modulates +/-30%% at 4-8 Hz with a per-texel phase (cheap shader noise, no extra texture lookups).", CVAR_ARCHIVE);

    // Camera controls.  Mouse-look engages while RIGHT mouse is held.
    PT_CVAR(cam_speed,         "3.0", "Movement speed (units/sec)",        CVAR_ARCHIVE);
    PT_CVAR(cam_sprint_mult,   "3.0", "Speed multiplier with shift",       CVAR_ARCHIVE);
    PT_CVAR(cam_sensitivity,   "0.12","Mouse-look sensitivity (deg/pixel)", CVAR_ARCHIVE);
    PT_CVAR(cam_fov,           "60.0","Vertical field of view (degrees)",  CVAR_ARCHIVE);
    PT_CVAR(cam_pos,           "0 1.5 4", "Camera position (x y z)",       CVAR_ARCHIVE);
    PT_CVAR(cam_yaw,           "0",       "Yaw in degrees",                CVAR_ARCHIVE);
    PT_CVAR(cam_pitch,         "-11.5",   "Pitch in degrees (clamped +/- 85)", CVAR_ARCHIVE);
    // Depth-of-field (thin-lens camera). Each primary ray originates
    // at a random sample on the aperture and aims through the focal-
    // plane intersection of the pinhole ray. Bokeh shape comes from
    // the aperture sampling distribution (round disk vs polygon).
    PT_CVAR(r_dof,             "0",   "Depth of field. 0 = pinhole camera (everything sharp). 1 = thin-lens with r_dof_aperture / r_dof_focal_distance.", CVAR_ARCHIVE);
    PT_CVAR(r_dof_aperture,    "0.05","Aperture radius in world units. Bigger = more blur on out-of-focus pixels. Real-camera analogue: focal_length / f_number; e.g. 50mm at f/2.8 ~= 0.018 (assuming the scene is in metres).", CVAR_ARCHIVE);
    PT_CVAR(r_dof_focal_distance, "5.0", "Distance from camera (world units) where the scene is in perfect focus. Closer / farther pixels get bokeh proportional to their distance from this plane.", CVAR_ARCHIVE);
    PT_CVAR(r_dof_blades,      "0",   "Aperture blade count. 0 = perfectly round disk (circular bokeh). 5/6/8 = polygonal iris (matching real lens aperture blades) -- gives polygonal bokeh on out-of-focus highlights.", CVAR_ARCHIVE);
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
        if (denoise_color_tex_id_    != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        if (depth_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
        if (motion_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
        if (post_denoise_hdr_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
        for (auto& id : bloom_mip_tex_id_) {
            if (id != 0) device_->DestroyTexture(pt::rhi::TextureHandle{id});
            id = 0;
        }
        if (bloom_dummy_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{bloom_dummy_tex_id_});
        if (env_map_tex_id_         != 0) device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
        if (env_marginal_cdf_id_    != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_marginal_cdf_id_});
        if (env_conditional_cdf_id_ != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_conditional_cdf_id_});
        if (star_map_tex_id_        != 0) device_->DestroyTexture(pt::rhi::TextureHandle{star_map_tex_id_});
    }
    scene_tlas_id_        = 0;
    box_blas_id_          = 0;
    box_vbuf_id_          = 0;
    box_ibuf_id_          = 0;
    prim_buffer_id_       = 0;
    prim_buffer_capacity_ = 0;
    denoise_color_tex_id_    = 0;
    depth_tex_id_            = 0;
    motion_tex_id_           = 0;
    post_denoise_hdr_tex_id_ = 0;
    tonemap_pipeline_id_     = 0;
    bloom_down_pipeline_id_  = 0;
    bloom_up_pipeline_id_    = 0;
    bloom_dummy_tex_id_      = 0;
    for (auto& id : bloom_mip_tex_id_) id = 0;
    for (auto& w  : bloom_mip_w_)      w  = 0;
    for (auto& h  : bloom_mip_h_)      h  = 0;
    env_map_tex_id_       = 0;
    env_marginal_cdf_id_      = 0;
    env_conditional_cdf_id_   = 0;
    env_total_luminance_      = 0.0f;
    star_map_tex_id_      = 0;
    star_map_present_     = 0;
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

    // Two compute kernels: the path tracer ("pathtrace") and the
    // post-denoise tonemap ("tonemap"). Both are pre-built at backend
    // init in MetalDevice; CreateComputePipeline just looks them up
    // by name and hands back a handle.
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "pathtrace", .bytecode = {}, .debug_name = "pathtrace",
        };
        pathtrace_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "tonemap", .bytecode = {}, .debug_name = "tonemap",
        };
        tonemap_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "bloom_down", .bytecode = {}, .debug_name = "bloom_down",
        };
        bloom_down_pipeline_id_ = device_->CreateComputePipeline(desc).id;
    }
    {
        pt::rhi::ComputePipelineDesc desc{
            .kernel_name = "bloom_up", .bytecode = {}, .debug_name = "bloom_up",
        };
        bloom_up_pipeline_id_ = device_->CreateComputePipeline(desc).id;
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

    // BSC starmap: load + rasterise once on this device.
    EnsureStarMapUploaded();
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

    auto destroy_env = [this]() {
        if (env_map_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{env_map_tex_id_});
        if (env_marginal_cdf_id_      != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_marginal_cdf_id_});
        if (env_conditional_cdf_id_   != 0) device_->DestroyBuffer(pt::rhi::BufferHandle{env_conditional_cdf_id_});
        env_map_tex_id_         = 0;
        env_marginal_cdf_id_    = 0;
        env_conditional_cdf_id_ = 0;
        env_total_luminance_    = 0.0f;
    };

    device_->WaitIdle();
    destroy_env();

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

    // Repack RGB -> RGBA + compute luminance per pixel for the CDF.
    const std::uint32_t W = img.width, H = img.height;
    std::vector<float> rgba(std::size_t(W) * H * 4);
    std::vector<float> conditional(std::size_t(W) * H);     // CDF per row
    std::vector<float> marginal(H);                          // CDF over rows
    double total = 0.0;
    for (std::uint32_t v = 0; v < H; ++v) {
        const double sin_theta = std::sin(M_PI * (double(v) + 0.5) / double(H));
        double row_sum = 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            const std::size_t pi = std::size_t(v) * W + u;
            float r = img.rgb[pi * 3 + 0];
            float g = img.rgb[pi * 3 + 1];
            float b = img.rgb[pi * 3 + 2];
            rgba[pi * 4 + 0] = r;
            rgba[pi * 4 + 1] = g;
            rgba[pi * 4 + 2] = b;
            rgba[pi * 4 + 3] = 1.0f;
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const double weight = double(lum) * sin_theta;
            row_sum += weight;
            conditional[pi] = float(row_sum);   // unnormalized prefix sum within row
        }
        // Normalize within row to [0, 1].
        const double norm = (row_sum > 0.0) ? (1.0 / row_sum) : 0.0;
        for (std::uint32_t u = 0; u < W; ++u) {
            conditional[std::size_t(v) * W + u] = float(double(conditional[std::size_t(v) * W + u]) * norm);
        }
        marginal[v] = float(row_sum);
        total += row_sum;
    }
    // Marginal: prefix-sum + normalize so marginal[H-1] == 1.0.
    {
        const double norm = (total > 0.0) ? (1.0 / total) : 0.0;
        double prefix = 0.0;
        for (std::uint32_t v = 0; v < H; ++v) {
            prefix += marginal[v];
            marginal[v] = float(prefix * norm);
        }
        if (H > 0) marginal[H - 1] = 1.0f;       // guard against fp drift
    }
    env_total_luminance_ = float(total);

    auto tex = device_->CreateTexture({
        .width  = W, .height = H,
        .format = pt::rhi::TextureFormat::RGBA32F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "env_map",
    });
    if (tex.id == 0 || !device_->WriteTexture(tex, rgba.data(), rgba.size() * sizeof(float))) {
        LOG_ERROR("env_map: texture create/upload failed ({}x{})", W, H);
        if (tex.id != 0) device_->DestroyTexture(tex);
        return;
    }
    env_map_tex_id_ = tex.id;

    auto m_buf = device_->CreateBuffer({
        .size = sizeof(float) * H, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "env_marginal_cdf",
    });
    auto c_buf = device_->CreateBuffer({
        .size = sizeof(float) * std::size_t(W) * H, .usage = pt::rhi::BufferUsage::Storage,
        .debug_name = "env_conditional_cdf",
    });
    if (m_buf.id == 0 || c_buf.id == 0) {
        LOG_ERROR("env_map: CDF buffer creation failed");
        device_->DestroyTexture(tex); env_map_tex_id_ = 0;
        if (m_buf.id != 0) device_->DestroyBuffer(m_buf);
        if (c_buf.id != 0) device_->DestroyBuffer(c_buf);
        return;
    }
    device_->WriteBuffer(m_buf, marginal.data(),    sizeof(float) * H);
    device_->WriteBuffer(c_buf, conditional.data(), sizeof(float) * std::size_t(W) * H);
    env_marginal_cdf_id_    = m_buf.id;
    env_conditional_cdf_id_ = c_buf.id;

    accum_dirty_ = true;
    LOG_INFO("env_map: loaded {} ({}x{} HDR, total luminance {:.2f})", path, W, H, env_total_luminance_);
}

void Engine::EnsureStarMapUploaded() {
    if (!device_) return;
    if (star_map_tex_id_ != 0) return;          // already uploaded on this device

    constexpr const char*  kPath = "assets/stars/BSC5.dat";
    constexpr std::uint32_t kW   = 4096;
    constexpr std::uint32_t kH   = 2048;

    std::string err;
    auto stars = pt::stars::LoadBsc5(kPath, &err);
    if (stars.empty()) {
        LOG_WARN("starmap: BSC load failed ({}); stars disabled", err);
        star_map_present_ = 0;
        return;
    }

    std::vector<float> rgba;
    pt::stars::RasteriseJ2000Map(stars, kW, kH, rgba);

    // GPU expects RGBA16F. The rasteriser hands us float-RGBA already in
    // half-friendly range (brightest entries ~25), so we let the RHI
    // convert on upload (Metal's WriteTexture for RGBA16F packs the
    // float32 source into half-precision). If the RHI can't accept
    // float32 source for an RGBA16F texture it will fail upload and
    // we'll fall back to no-stars.
    auto tex = device_->CreateTexture({
        .width  = kW, .height = kH,
        .format = pt::rhi::TextureFormat::RGBA16F,
        .usage  = pt::rhi::TextureUsage::Storage,
        .debug_name = "star_map_j2000",
    });
    if (tex.id == 0) {
        LOG_WARN("starmap: texture create failed; stars disabled");
        star_map_present_ = 0;
        return;
    }
    // RGBA16F upload: the Metal RHI's WriteTexture path expects bytes
    // matching the texture format. Pack to half manually (cheap).
    std::vector<std::uint16_t> half(rgba.size());
    auto f32_to_f16 = [](float f) -> std::uint16_t {
        // Standard IEEE-754 half conversion. Branchless path for the
        // normalised range, with a flush-to-zero fallback for tiny
        // values; star fluxes never overflow half max (~65504).
        std::uint32_t u; std::memcpy(&u, &f, sizeof(u));
        std::uint32_t sign = (u >> 16) & 0x8000;
        std::int32_t  exp  = static_cast<std::int32_t>((u >> 23) & 0xFF) - 127 + 15;
        std::uint32_t mant = u & 0x7FFFFF;
        if (exp <= 0)  return static_cast<std::uint16_t>(sign);                 // underflow -> 0
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00);        // overflow -> inf
        return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
    };
    for (std::size_t i = 0; i < rgba.size(); ++i) half[i] = f32_to_f16(rgba[i]);
    if (!device_->WriteTexture(tex, half.data(), half.size() * sizeof(std::uint16_t))) {
        LOG_WARN("starmap: texture upload failed; stars disabled");
        device_->DestroyTexture(tex);
        star_map_present_ = 0;
        return;
    }
    star_map_tex_id_  = tex.id;
    star_map_present_ = 1;
    // Diagnostics: how much of the rasterised map actually has flux?
    // If nonzero count is low or peak is tiny something has gone wrong
    // in the splatter (clamps, magnitude scaling, etc.).
    {
        std::size_t nonzero = 0;
        float peak = 0.0f;
        const std::size_t N = std::size_t(kW) * kH;
        for (std::size_t i = 0; i < N; ++i) {
            float r = rgba[i * 4 + 0];
            float g = rgba[i * 4 + 1];
            float b = rgba[i * 4 + 2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (lum > 0.0f) ++nonzero;
            if (lum > peak) peak = lum;
        }
        LOG_INFO("starmap: loaded {} stars (BSC5), {}x{} J2000 RGBA16F "
                 "(nonzero texels {}/{}, peak luminance {:.3f})",
                 stars.size(), kW, kH, nonzero, N, peak);
    }
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
            if (denoise_color_tex_id_    != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
            if (depth_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
            if (motion_tex_id_           != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
            if (post_denoise_hdr_tex_id_ != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
            for (auto& id : bloom_mip_tex_id_) {
                if (id != 0) device_->DestroyTexture(pt::rhi::TextureHandle{id});
                id = 0;
            }
            denoise_color_tex_id_ = depth_tex_id_ = motion_tex_id_ = post_denoise_hdr_tex_id_ = 0;
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
    // denoising is off. The post-denoise HDR intermediate is the linear
    // RGBA16F MetalFX writes to; the `tonemap` compute kernel reads it
    // and writes the swapchain.
    if (denoiser_active_ &&
        (denoise_color_tex_id_ == 0 || depth_tex_id_ == 0 || motion_tex_id_ == 0 ||
         post_denoise_hdr_tex_id_ == 0 || size_changed)) {
        if (denoise_color_tex_id_     != 0) device_->DestroyTexture(pt::rhi::TextureHandle{denoise_color_tex_id_});
        if (depth_tex_id_             != 0) device_->DestroyTexture(pt::rhi::TextureHandle{depth_tex_id_});
        if (motion_tex_id_            != 0) device_->DestroyTexture(pt::rhi::TextureHandle{motion_tex_id_});
        if (post_denoise_hdr_tex_id_  != 0) device_->DestroyTexture(pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
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
        auto post_h = device_->CreateTexture({
            .width = fc.width, .height = fc.height,
            .format = pt::rhi::TextureFormat::RGBA16F,
            .usage  = pt::rhi::TextureUsage::Storage,
            .debug_name = "post_denoise_hdr",
        });
        denoise_color_tex_id_    = color_h.id;
        depth_tex_id_            = depth_h.id;
        motion_tex_id_           = motion_h.id;
        post_denoise_hdr_tex_id_ = post_h.id;
        prev_view_proj_valid_ = false;        // history is invalid after resize
        if (denoise_color_tex_id_ == 0 || depth_tex_id_ == 0 ||
            motion_tex_id_        == 0 || post_denoise_hdr_tex_id_ == 0) {
            LOG_ERROR("denoiser G-buffer allocation failed at {}x{}", fc.width, fc.height);
            denoiser_active_ = false;
        }
        // Bloom mip chain. mip[0] is half-res; each subsequent mip
        // halves again. Caps at 1x1 if the swapchain is tiny. Always
        // allocated alongside the denoiser textures since bloom is
        // bound through the tonemap pass.
        for (int i = 0; i < kBloomMips; ++i) {
            if (bloom_mip_tex_id_[i] != 0) {
                device_->DestroyTexture(pt::rhi::TextureHandle{bloom_mip_tex_id_[i]});
                bloom_mip_tex_id_[i] = 0;
            }
            std::uint32_t bw = std::max(1u, fc.width  >> (i + 1));
            std::uint32_t bh = std::max(1u, fc.height >> (i + 1));
            auto bm = device_->CreateTexture({
                .width = bw, .height = bh,
                .format = pt::rhi::TextureFormat::RGBA16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "bloom_mip",
            });
            bloom_mip_tex_id_[i] = bm.id;
            bloom_mip_w_[i]      = bw;
            bloom_mip_h_[i]      = bh;
        }
        // 1x1 zero-filled placeholder bound to the tonemap pass when
        // bloom is disabled. Tonemap's bloom_intensity push is set to
        // 0 in that case, but Metal still demands the slot be bound;
        // a tiny dummy avoids leaving the slot empty.
        if (bloom_dummy_tex_id_ == 0) {
            auto dh = device_->CreateTexture({
                .width = 1, .height = 1,
                .format = pt::rhi::TextureFormat::RGBA16F,
                .usage  = pt::rhi::TextureUsage::Storage,
                .debug_name = "bloom_dummy",
            });
            bloom_dummy_tex_id_ = dh.id;
            std::uint16_t zero[4] {0,0,0,0};
            if (dh.id != 0) device_->WriteTexture(dh, zero, sizeof(zero));
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
    // P11 MIS CDFs at MSL slots 4/5. CRITICAL: always bind something
    // here even if no env map is loaded -- the shader declares these
    // buffers, so leaving slots 4/5 unbound shifts the dynamically-
    // assigned push-constant slot to the wrong index (Metal computes
    // it as max-bound + 1) and the shader reads garbage push fields
    // -> all-black render. When env CDFs aren't available we re-use
    // the prim_buffer as a harmless placeholder; the shader never
    // reads from slots 4/5 when env_map_present == 0.
    pt::rhi::BufferHandle slot4 = (env_marginal_cdf_id_ != 0)
        ? pt::rhi::BufferHandle{env_marginal_cdf_id_}
        : pt::rhi::BufferHandle{prim_buffer_id_};
    pt::rhi::BufferHandle slot5 = (env_conditional_cdf_id_ != 0)
        ? pt::rhi::BufferHandle{env_conditional_cdf_id_}
        : pt::rhi::BufferHandle{prim_buffer_id_};
    if (slot4.id != 0) cb->BindBuffer(4, slot4, 0);
    if (slot5.id != 0) cb->BindBuffer(5, slot5, 0);
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
    // BSC starmap (always bind when present so the shader's binding is
    // satisfied; sampling is gated by star_map_present in the push so
    // the texture being a black 1x1 placeholder is not a problem).
    if (star_map_tex_id_ != 0) {
        cb->BindStorageTexture(6, pt::rhi::TextureHandle{star_map_tex_id_});
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
        float env_total_luminance;
        float curr_view_proj[16];
        float prev_view_proj[16];
        float sun_and_mode[4];            // .xyz = sun_dir, .w = float(sky_mode)
        float exposure_pad[4];            // .x = exposure, .y = procedural-stars on, .z = bsc-stars on
        // World->J2000 rotation, row-major. Each row stored as float4 so
        // Slang's std430 layout (which pads vec3 to vec4 anyway) lines up
        // with the host. .w of each row is unused.
        float w2j_row0[4];
        float w2j_row1[4];
        float w2j_row2[4];
        // .x = aperture radius (0 = pinhole, no DOF). .y = focal
        // distance (world units). .z = aperture blade count (0 =
        // round disk, 3..16 = polygonal iris). .w reserved.
        float dof_params[4];
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
    push.env_total_luminance = env_total_luminance_;

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

    // Sun direction from elevation/azimuth (degrees). Convention:
    // +Y up, -Z forward (north), +X right (east). When
    // r_sky_use_astronomical is on, override with computed positions
    // from the observer lat/lon and current UTC.
    float sun_elev_deg = 30.0f, sun_azim_deg = 135.0f;
    bool astro_on = false;
    if (auto* v = C.FindCVar("r_sky_use_astronomical")) astro_on = v->GetBool();
    // Compute JD from r_sky_hour (UTC hour of day on today's date). The
    // cvar is the only time knob -- animation just drives this value
    // forward over wall-clock seconds.
    auto compute_jd = [&]() -> double {
        float hour = 12.0f;
        if (auto* v = C.FindCVar("r_sky_hour")) hour = v->GetFloat();
        bool local = true;
        if (auto* v = C.FindCVar("r_sky_hour_local")) local = v->GetBool();
        float tz = 0.0f;
        if (local) {
            if (auto* v = C.FindCVar("r_sky_tz_offset_hours")) tz = v->GetFloat();
        }
        // r_sky_hour interpreted in local civil time -> subtract the
        // city's UTC offset to get UTC hour. May go negative or above
        // 24 if the user's local hour straddles a UTC day boundary;
        // dividing by 24 below handles that automatically.
        const float hour_utc = hour - tz;
        const std::time_t now = std::time(nullptr);
        const double jd_inst = pt::astro::julianDateFromTimeT(now);
        // JD's day rolls over at noon UTC; floor(jd-0.5)+0.5 is the
        // UTC midnight that opened the *current* civil day.
        return std::floor(jd_inst - 0.5) + 0.5 + double(hour_utc) / 24.0;
    };
    if (astro_on) {
        double lat = 13.0827, lon = 80.2707;
        if (auto* v = C.FindCVar("r_sky_lat")) lat = v->GetFloat();
        if (auto* v = C.FindCVar("r_sky_lon")) lon = v->GetFloat();
        const double jd = compute_jd();
        auto sun_eq = pt::astro::sunPosition(jd);
        auto sun_h  = pt::astro::equatorialToHorizon(sun_eq, lat, lon, jd);
        sun_elev_deg = static_cast<float>(sun_h.altitude_deg);
        sun_azim_deg = static_cast<float>(sun_h.azimuth_deg);
        // Reflect computed values back to the manual cvars so the
        // settings panel shows the real numbers (read-only feel).
        C.SetCVarOverride("r_sun_elevation", std::to_string(sun_elev_deg));
        C.SetCVarOverride("r_sun_azimuth",   std::to_string(sun_azim_deg));
    } else {
        if (auto* v = C.FindCVar("r_sun_elevation")) sun_elev_deg = v->GetFloat();
        if (auto* v = C.FindCVar("r_sun_azimuth"))   sun_azim_deg = v->GetFloat();
    }
    const float elev_r = glm::radians(sun_elev_deg);
    const float azim_r = glm::radians(sun_azim_deg);
    const float ce = std::cos(elev_r), se = std::sin(elev_r);
    push.sun_and_mode[0] =  ce * std::sin(azim_r);
    push.sun_and_mode[1] =  se;
    push.sun_and_mode[2] = -ce * std::cos(azim_r);

    // Sky mode resolution. "hdri" with no env map loaded falls back
    // to gradient so the shader doesn't read an unbound texture.
    std::string sky_mode_str = "procedural";
    if (auto* v = C.FindCVar("r_sky_mode")) sky_mode_str = v->value;
    std::uint32_t sky_mode_id = 2;
    if      (sky_mode_str == "gradient")   sky_mode_id = 0;
    else if (sky_mode_str == "hdri")       sky_mode_id = (env_map_tex_id_ != 0) ? 1u : 0u;
    else /* procedural (default) */        sky_mode_id = 2;
    push.sun_and_mode[3] = float(sky_mode_id);
    push.env_map_present = (sky_mode_id == 1u) ? 1u : 0u;

    bool auto_exp = false;
    if (auto* v = C.FindCVar("r_auto_exposure")) auto_exp = v->GetBool();
    float manual_exp = 1.5f;
    if (auto* v = C.FindCVar("r_exposure")) manual_exp = v->GetFloat();
    push.exposure_pad[0] = auto_exp ? current_exposure_ : manual_exp;
    bool show_stars = true;
    if (auto* v = C.FindCVar("r_show_stars")) show_stars = v->GetBool();
    std::string stars_mode = "bsc";
    if (auto* v = C.FindCVar("r_stars_mode")) stars_mode = v->value;
    // BSC requested + catalog loaded -> sample the J2000 starmap (.z).
    // Otherwise procedural hash starfield (.y). r_show_stars=0 zeroes both.
    const bool want_bsc =
        show_stars && stars_mode == "bsc" && star_map_present_ != 0u;
    const bool want_procedural =
        show_stars && (stars_mode == "procedural" ||
                       (stars_mode == "bsc" && star_map_present_ == 0u));
    push.exposure_pad[1] = want_procedural ? 1.0f : 0.0f;
    push.exposure_pad[2] = want_bsc        ? 1.0f : 0.0f;
    bool twinkle = true;
    if (auto* v = C.FindCVar("r_stars_twinkle")) twinkle = v->GetBool();
    push.exposure_pad[3] = twinkle ? 1.0f : 0.0f;

    // World->J2000 rotation. Always computed; the shader scales the
    // starmap sample by exposure_pad.z so when stars are off the
    // matrix is just unused work. lat/lon come from the geographic
    // cvars even when r_sky_use_astronomical = 0 -- stars still need
    // the observer's actual location to be in the right place.
    {
        double lat = 13.0827, lon = 80.2707;
        if (auto* v = C.FindCVar("r_sky_lat")) lat = v->GetFloat();
        if (auto* v = C.FindCVar("r_sky_lon")) lon = v->GetFloat();
        const double jd = compute_jd();
        float m[9];
        pt::astro::worldToJ2000Matrix(lat, lon, jd, m);
        push.w2j_row0[0] = m[0]; push.w2j_row0[1] = m[1]; push.w2j_row0[2] = m[2];
        push.w2j_row1[0] = m[3]; push.w2j_row1[1] = m[4]; push.w2j_row1[2] = m[5];
        push.w2j_row2[0] = m[6]; push.w2j_row2[1] = m[7]; push.w2j_row2[2] = m[8]; push.w2j_row2[3] = 0.0f;
        // The 3x3 rotation only fills 9 floats; the w-lanes are
        // available scratch. Pack engine flags here:
        //   row0.w = HDR-pipeline (1 = raw HDR through MetalFX).
        //   row1.w = refractive-shadow-ray budget. 0 disables caustics
        //     entirely (NEE rays treat dielectrics as opaque); positive
        //     = max number of dielectric bounces a shadow ray may chain
        //     before giving up. Stored as a float because Slang push
        //     constants are easier to align that way.
        bool hdr_pipeline = true;
        if (auto* v = C.FindCVar("r_hdr_pipeline")) hdr_pipeline = v->GetBool();
        push.w2j_row0[3] = hdr_pipeline ? 1.0f : 0.0f;
        bool caustics = true;
        if (auto* v = C.FindCVar("r_caustics")) caustics = v->GetBool();
        int refract_bounces = 4;
        if (auto* v = C.FindCVar("r_refract_bounces")) refract_bounces = v->GetInt();
        if (refract_bounces < 0)  refract_bounces = 0;
        if (refract_bounces > 16) refract_bounces = 16;
        push.w2j_row1[3] = caustics ? float(refract_bounces) : 0.0f;
    }

    {
        bool dof_on = false;
        if (auto* v = C.FindCVar("r_dof")) dof_on = v->GetBool();
        float aperture = 0.0f, focal_dist = 5.0f, blades = 0.0f;
        if (auto* v = C.FindCVar("r_dof_aperture"))        aperture   = v->GetFloat();
        if (auto* v = C.FindCVar("r_dof_focal_distance"))  focal_dist = v->GetFloat();
        if (auto* v = C.FindCVar("r_dof_blades"))          blades     = float(v->GetInt());
        push.dof_params[0] = dof_on ? aperture : 0.0f;
        push.dof_params[1] = focal_dist;
        push.dof_params[2] = blades;
        push.dof_params[3] = 0.0f;
    }

    static_assert(sizeof(PtPush) == 272 + 48 + 16);
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
        // MetalFX writes to the linear-HDR intermediate; the tonemap
        // dispatch below converts that to sRGB and writes the swapchain.
        dd.output        = pt::rhi::TextureHandle{post_denoise_hdr_tex_id_};
        dd.jitter_x      = last_jitter_x_;
        dd.jitter_y      = last_jitter_y_;
        dd.reset_history = !prev_view_proj_valid_;
        // MetalFX wants worldToView and viewToClip separately, so pass
        // them rather than the combined view*proj. glm matrices are
        // column-major, matching simd_float4x4 on the consumer end.
        dd.world_to_view = glm::value_ptr(view);
        dd.view_to_clip  = glm::value_ptr(proj);
        device_->Denoise(dd);

        // Bloom pyramid: extract bright HDR pixels into bloom_mip[0]
        // (with luminance threshold), then progressively downsample
        // through the chain, then upsample additively back up to mip
        // 0. The result in mip 0 is sampled by the tonemap kernel
        // and added pre-ACES so the bloom gets the same curve squash
        // as the rest of the image.
        bool bloom_on = false;
        if (auto* v = C.FindCVar("r_bloom")) bloom_on = v->GetBool();
        float bloom_thresh = 1.0f, bloom_intensity = 0.05f;
        int   bloom_mips   = kBloomMips;
        if (auto* v = C.FindCVar("r_bloom_threshold")) bloom_thresh    = v->GetFloat();
        if (auto* v = C.FindCVar("r_bloom_intensity")) bloom_intensity = v->GetFloat();
        if (auto* v = C.FindCVar("r_bloom_mips"))      bloom_mips      = v->GetInt();
        if (bloom_mips < 1) bloom_mips = 1;
        if (bloom_mips > kBloomMips) bloom_mips = kBloomMips;

        if (bloom_on && bloom_mip_tex_id_[0] != 0 && bloom_intensity > 0.0f) {
            // Downsample chain: post_denoise_hdr -> mip0 (with
            // threshold), mip0 -> mip1, ..., mip[N-2] -> mip[N-1].
            for (int i = 0; i < bloom_mips; ++i) {
                cb->BindComputePipeline(pt::rhi::PipelineHandle{bloom_down_pipeline_id_});
                pt::rhi::TextureHandle src_h{
                    (i == 0) ? post_denoise_hdr_tex_id_ : bloom_mip_tex_id_[i - 1]};
                pt::rhi::TextureHandle dst_h{bloom_mip_tex_id_[i]};
                cb->BindStorageTexture(0, src_h);
                cb->BindStorageTexture(1, dst_h);
                struct DownPush { float threshold; float pad[3]; } dp{};
                // Threshold only applies on the first mip; later mips
                // are downsampling already-extracted bright pixels.
                dp.threshold = (i == 0) ? bloom_thresh : 0.0f;
                cb->PushConstants(&dp, sizeof(dp));
                cb->Dispatch((bloom_mip_w_[i] + 7) / 8,
                             (bloom_mip_h_[i] + 7) / 8, 1);
            }
            // Upsample chain: mip[N-1] -> mip[N-2] (additive),
            // mip[N-2] -> mip[N-3], ..., mip 1 -> mip 0. Result
            // accumulates into mip 0 which is the layer the tonemap
            // pass samples. r_bloom_radius widens the per-mip
            // sample spread for a softer halo.
            float bloom_radius = 1.0f;
            if (auto* v = C.FindCVar("r_bloom_radius")) bloom_radius = v->GetFloat();
            for (int i = bloom_mips - 1; i > 0; --i) {
                cb->BindComputePipeline(pt::rhi::PipelineHandle{bloom_up_pipeline_id_});
                cb->BindStorageTexture(0, pt::rhi::TextureHandle{bloom_mip_tex_id_[i]});
                cb->BindStorageTexture(1, pt::rhi::TextureHandle{bloom_mip_tex_id_[i - 1]});
                struct UpPush { float radius; float pad[3]; } up{};
                up.radius = bloom_radius;
                cb->PushConstants(&up, sizeof(up));
                cb->Dispatch((bloom_mip_w_[i - 1] + 7) / 8,
                             (bloom_mip_h_[i - 1] + 7) / 8, 1);
            }
        }

        // Post-denoise tonemap: linear HDR -> exposure -> ACES -> sRGB
        // swapchain (gamma encode is implicit on store of the BGRA8_sRGB
        // surface).
        cb->BindComputePipeline(pt::rhi::PipelineHandle{tonemap_pipeline_id_});
        cb->BindStorageTexture(0, pt::rhi::TextureHandle{post_denoise_hdr_tex_id_});
        cb->BindStorageTexture(1, fc.swapchain_image);
        // Bind bloom mip 0 (built above) when bloom is on, else a
        // 1x1 zero texture so the slot has *something* and the
        // shader's branch on bloom_intensity > 0 keeps it skipped.
        pt::rhi::TextureHandle bloom_h{
            (bloom_on && bloom_intensity > 0.0f && bloom_mip_tex_id_[0] != 0)
                ? bloom_mip_tex_id_[0]
                : bloom_dummy_tex_id_};
        if (bloom_h.id != 0) cb->BindStorageTexture(2, bloom_h);
        bool hdr_pipeline = true;
        if (auto* v = C.FindCVar("r_hdr_pipeline")) hdr_pipeline = v->GetBool();
        bool flare_on = false;
        float flare_int = 0.15f, flare_disp = 0.012f, flare_thresh = 3.0f, flare_size = 0.10f;
        int   flare_count = 4;
        std::string flare_mode = "sun";
        if (auto* v = C.FindCVar("r_lens_flare"))            flare_on     = v->GetBool();
        if (auto* v = C.FindCVar("r_lens_flare_intensity"))  flare_int    = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_dispersion")) flare_disp   = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_count"))      flare_count  = v->GetInt();
        if (auto* v = C.FindCVar("r_lens_flare_threshold"))  flare_thresh = v->GetFloat();
        if (auto* v = C.FindCVar("r_lens_flare_mode"))       flare_mode   = v->value;
        if (auto* v = C.FindCVar("r_lens_flare_size"))       flare_size   = v->GetFloat();
        if (flare_count < 1) flare_count = 1;
        if (flare_count > 6) flare_count = 6;

        // Project sun direction to screen-space UV. Used by 'sun'
        // mode to draw discrete ghost discs at the sun's mirrored
        // screen position. Sentinel (-2, -2) means sun is behind
        // the camera or the cvars are missing -- shader skips the
        // flare loop in that case.
        float sun_uv_x = -2.0f, sun_uv_y = -2.0f;
        {
            const float sx = push.sun_and_mode[0];
            const float sy = push.sun_and_mode[1];
            const float sz = push.sun_and_mode[2];
            const glm::vec3 sun_world{sx, sy, sz};
            const float fwd_dot = glm::dot(sun_world, fwd);
            if (fwd_dot > 1e-3f) {
                const float right_dot = glm::dot(sun_world, right);
                const float up_dot    = glm::dot(sun_world, up);
                const float ft = cam.FovYTan();
                const float xs = (right_dot / fwd_dot) / (ft * aspect);
                const float ys = (up_dot    / fwd_dot) / ft;
                sun_uv_x = xs * 0.5f + 0.5f;
                sun_uv_y = -ys * 0.5f + 0.5f;
            }
        }

        struct TonePush {
            float        exposure;
            std::uint32_t passthrough;
            float        bloom_intensity;
            float        flare_intensity;
            float        flare_dispersion;
            std::uint32_t flare_count;
            float        flare_threshold;
            std::uint32_t flare_mode_sun;   // 1 = explicit sun-position flare
            float        flare_size;        // sun mode: ghost disc base radius
            float        sun_uv[2];         // sun's screen UV
            float        pad;
        } tp{};
        tp.exposure         = push.exposure_pad[0];
        tp.passthrough      = hdr_pipeline ? 0u : 1u;
        tp.bloom_intensity  = (bloom_on && bloom_h.id == bloom_mip_tex_id_[0])
                                ? bloom_intensity : 0.0f;
        // Flare also samples the bloom mip, so it depends on bloom
        // having actually run. If bloom is off the bloom_tex slot is
        // the 1x1 placeholder and the flare loop would just sample
        // zeros -- gate it explicitly to skip the loop entirely.
        const bool bloom_ran = bloom_on && bloom_h.id == bloom_mip_tex_id_[0];
        // 'sun' mode draws explicit ghosts (no bloom needed); 'image'
        // mode samples bloom and therefore needs bloom_ran.
        const bool flare_sun_mode = (flare_mode == "sun");
        tp.flare_intensity  = (flare_on && (flare_sun_mode || bloom_ran))
                                ? flare_int : 0.0f;
        tp.flare_dispersion = flare_disp;
        tp.flare_count      = static_cast<std::uint32_t>(flare_count);
        tp.flare_threshold  = flare_thresh;
        tp.flare_mode_sun   = flare_sun_mode ? 1u : 0u;
        tp.flare_size       = flare_size;
        tp.sun_uv[0]        = sun_uv_x;
        tp.sun_uv[1]        = sun_uv_y;
        cb->PushConstants(&tp, sizeof(tp));
        cb->Dispatch((fc.width + 7) / 8, (fc.height + 7) / 8, 1);
    }

    device_->Submit(cb);
    device_->EndFrame(cb);

    prev_view_proj_       = curr_view_proj;
    prev_view_proj_valid_ = true;

    // Auto-exposure: every 8 frames, sample accum_hdr's central region
    // and nudge the exposure scalar toward 0.18 / scene_avg_luminance.
    // accum_hdr is shared-storage on Apple Silicon so the readback is
    // a memcpy; we step every 16th pixel for speed (~3.6k samples on a
    // 1280x720 frame). Smoothing factor controls eye-adaptation rate.
    if (auto_exp && accum_texture_id_ != 0 && ++autoexpose_counter_ >= 8) {
        autoexpose_counter_ = 0;
        std::uint32_t w = 0, h = 0;
        const std::size_t bytes = std::size_t(accum_w_) * accum_h_ * 16;
        std::vector<float> rgba(bytes / sizeof(float));
        if (device_->ReadbackTexture(pt::rhi::TextureHandle{accum_texture_id_},
                                     rgba.data(), bytes, &w, &h) && w > 0 && h > 0) {
            double accum_lum = 0.0;
            std::size_t n = 0;
            for (std::uint32_t y = 0; y < h; y += 16) {
                for (std::uint32_t x = 0; x < w; x += 16) {
                    const float* p = rgba.data() + (std::size_t(y) * w + x) * 4;
                    const float lum = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
                    if (lum > 0.0f) { accum_lum += lum; ++n; }
                }
            }
            if (n > 0) {
                const float avg_lum = float(accum_lum / double(n));
                float key      = 0.18f;
                float exp_min  = 0.05f;
                float exp_max  = 4.0f;
                float k        = 0.20f;
                auto& Cx = pt::console::Console::Get();
                if (auto* v = Cx.FindCVar("r_exposure_target")) key     = v->GetFloat();
                if (auto* v = Cx.FindCVar("r_exposure_min"))    exp_min = v->GetFloat();
                if (auto* v = Cx.FindCVar("r_exposure_max"))    exp_max = v->GetFloat();
                if (auto* v = Cx.FindCVar("r_eye_adapt_speed")) k       = v->GetFloat();
                float target_exp = key / std::max(avg_lum, 1e-3f);
                if (target_exp < exp_min) target_exp = exp_min;
                if (target_exp > exp_max) target_exp = exp_max;
                // Geometric smoothing in log space gives a more
                // perceptually uniform fade than a linear lerp.
                if (k <= 0.0f) {
                    // Locked-exposure presets (DSLR / linear): jump
                    // straight to target without smoothing so the
                    // user sees the locked value immediately.
                    current_exposure_ = target_exp;
                } else {
                    current_exposure_ = std::exp(
                        std::log(current_exposure_) * (1.0f - k) +
                        std::log(target_exp)        * k);
                }
            }
        }
    }
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

    // Sky animation: advance r_sky_hour by `rate * dt` real-time
    // seconds. Wraps modulo 24. Marks accumulation dirty so the path
    // tracer drops its history each frame the time is actually moving.
    {
        auto& C = pt::console::Console::Get();
        bool animate = false;
        if (auto* v = C.FindCVar("r_sky_animate")) animate = v->GetBool();
        if (animate) {
            float rate = 0.0f;
            if (auto* v = C.FindCVar("r_sky_animate_rate")) rate = v->GetFloat();
            float hour = 12.0f;
            if (auto* v = C.FindCVar("r_sky_hour")) hour = v->GetFloat();
            hour += rate * float(dt);
            // Wrap into [0, 24).
            hour = std::fmod(hour, 24.0f);
            if (hour < 0.0f) hour += 24.0f;
            C.SetCVarOverride("r_sky_hour", std::to_string(hour));
            accum_dirty_ = true;
        }
    }

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

    // Auto-focus: trace a ray through the screen centre and write the
    // hit distance into r_dof_focal_distance. Like tap-to-focus on a
    // phone camera. CPU-side intersection against the analytic
    // primitives only -- mesh hits aren't checked yet (would need a
    // CPU BVH or a GPU depth_tex read-back). In practice the default
    // scene uses analytic primitives so this works for the common
    // case; extend later when mesh-aware focus matters.
    C.RegisterCommand("dump_sun_uv",
        "Print where the engine thinks the sun lands on screen "
        "(used by r_lens_flare_mode = sun). Sentinel like (-2, -2) "
        "means the sun is behind the camera.",
        [this](auto, pt::console::Output& out) {
            if (!camera_) { out.PrintLine("no camera"); return; }
            const auto& cam = *camera_;
            const glm::vec3 fwd   = cam.Forward();
            const glm::vec3 right = cam.Right();
            const glm::vec3 up    = cam.Up();
            auto& C2 = pt::console::Console::Get();
            float elev = 0.0f, azim = 0.0f;
            if (auto* v = C2.FindCVar("r_sun_elevation")) elev = v->GetFloat();
            if (auto* v = C2.FindCVar("r_sun_azimuth"))   azim = v->GetFloat();
            float er = glm::radians(elev), ar = glm::radians(azim);
            glm::vec3 sun{
                std::cos(er) * std::sin(ar),
                std::sin(er),
                -std::cos(er) * std::cos(ar),
            };
            float fwd_dot = glm::dot(sun, fwd);
            float ft = cam.FovYTan();
            float aspect = 16.0f / 9.0f;
            if (auto* v = C2.FindCVar("app_window_width")) {
                int w = v->GetInt();
                if (auto* h = C2.FindCVar("app_window_height")) {
                    int hh = h->GetInt();
                    if (hh > 0) aspect = float(w) / float(hh);
                }
            }
            if (fwd_dot <= 1e-3f) {
                out.FormatLine("sun is behind/sideways: fwd_dot={:.3f} (sentinel)",
                               fwd_dot);
                return;
            }
            float xs = (glm::dot(sun, right) / fwd_dot) / (ft * aspect);
            float ys = (glm::dot(sun, up)    / fwd_dot) / ft;
            float ux = xs * 0.5f + 0.5f;
            float uy = -ys * 0.5f + 0.5f;
            out.FormatLine("sun_dir=({:.3f},{:.3f},{:.3f}) "
                           "fwd_dot={:.3f} screen_uv=({:.3f},{:.3f}) {}",
                           sun.x, sun.y, sun.z, fwd_dot, ux, uy,
                           (ux >= 0 && ux <= 1 && uy >= 0 && uy <= 1)
                               ? "(on-screen)" : "(off-screen)");
        });

    C.RegisterCommand("dof_focus_here",
        "Auto-focus DOF on whatever's at the centre of the screen. "
        "Writes the hit distance into r_dof_focal_distance and turns "
        "r_dof on if it wasn't already.",
        [this](auto args, pt::console::Output& out) {
            (void)args;
            if (!camera_) { out.PrintLine("no camera"); return; }
            const auto& cam = *camera_;
            const glm::vec3 ro = cam.pos;
            const glm::vec3 rd = cam.Forward();   // already unit length

            float best_t = 1e30f;
            for (const auto& [id, p] : primitives_) {
                float t = best_t;
                if (p.type == AnalyticPrim::Sphere) {
                    // Standard ray-sphere intersection.
                    glm::vec3 c{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
                    glm::vec3 oc = ro - c;
                    float b = glm::dot(oc, rd);
                    float disc = b * b - (glm::dot(oc, oc) - p.radius_or_d * p.radius_or_d);
                    if (disc < 0.0f) continue;
                    float sq = std::sqrt(disc);
                    float t0 = -b - sq;
                    float t1 = -b + sq;
                    t = (t0 > 1e-3f) ? t0 : ((t1 > 1e-3f) ? t1 : best_t);
                } else { // Plane
                    glm::vec3 n{p.pos_or_n[0], p.pos_or_n[1], p.pos_or_n[2]};
                    float ndotd = glm::dot(n, rd);
                    if (std::fabs(ndotd) < 1e-6f) continue;
                    float th = -(glm::dot(n, ro) + p.radius_or_d) / ndotd;
                    if (th > 1e-3f) t = th;
                }
                if (t < best_t) best_t = t;
            }

            if (best_t >= 1e29f) {
                out.PrintLine("dof_focus_here: nothing in the centre of the "
                              "frame (no analytic primitive hit). Mesh hits "
                              "aren't checked yet.");
                return;
            }

            auto& C2 = pt::console::Console::Get();
            C2.SetCVarOverride("r_dof_focal_distance", std::to_string(best_t));
            C2.SetCVarOverride("r_dof", "1");
            accum_dirty_ = true;
            out.FormatLine("dof_focus_here: focal distance set to {:.3f} (r_dof on)",
                           best_t);
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
    if (auto* v = C.FindCVar("r_hdr_pipeline")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_bloom")) {
        v->allowed_values = {"0", "1"};
    }
    if (auto* v = C.FindCVar("r_lens_flare")) {
        v->allowed_values = {"0", "1"};
    }
    if (auto* v = C.FindCVar("r_lens_flare_mode")) {
        v->allowed_values = {"sun", "image"};
    }
    // Bloom + flare intensity / threshold / mips / count / dispersion
    // don't reset accumulation: they're applied in the post-tonemap
    // pass each frame, no dependency on path-tracer state.
    if (auto* v = C.FindCVar("r_caustics")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_refract_bounces")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_aperture")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_focal_distance")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_dof_blades")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // r_quality: master preset that bulk-edits the per-feature cvars.
    // Ranges chosen so 'low' is comfortably fast at 1080p on M-series
    // GPUs, 'high' is the headline-correct path, and 'ultra' bumps
    // sample counts past convergence + max refraction depth for static
    // showcase shots. 'custom' is the escape hatch that leaves the
    // sub-cvars whatever the user / autoexec last set.
    if (auto* v = C.FindCVar("r_quality")) {
        v->allowed_values = {"low", "medium", "high", "ultra", "custom"};
        v->on_change = [this](const pt::console::CVar& cv) {
            struct QPreset {
                const char* name;
                int spp, max_bounces, refract_bounces;
                bool caustics;
            };
            static const QPreset presets[] = {
                // SPP / bounces tuned for M4 Max @ 1080p; soft denoiser
                // already covers the noise on top.
                {"low",    1,  3, 1, false},  // no caustics, single shadow ray
                {"medium", 1,  6, 2, true},   // caustics on, modest depth
                {"high",   1,  8, 4, true},   // current default
                {"ultra",  4, 16, 8, true},   // showcase / stills
            };
            if (cv.value == "custom") { accum_dirty_ = true; return; }
            for (const auto& p : presets) {
                if (cv.value == p.name) {
                    auto& C2 = pt::console::Console::Get();
                    C2.SetCVarOverride("r_spp",             std::to_string(p.spp));
                    C2.SetCVarOverride("r_max_bounces",     std::to_string(p.max_bounces));
                    C2.SetCVarOverride("r_refract_bounces", std::to_string(p.refract_bounces));
                    C2.SetCVarOverride("r_caustics",        p.caustics ? "1" : "0");
                    accum_dirty_ = true;
                    LOG_INFO("r_quality: '{}' -> spp={} bounces={} refract={} caustics={}",
                             p.name, p.spp, p.max_bounces, p.refract_bounces,
                             p.caustics ? 1 : 0);
                    return;
                }
            }
        };
    }
    // r_eye_model: presets that shove sensible defaults into the
    // exposure clamp / target / adapt-speed cvars. Picked so the
    // visual differences are obvious: cats see further into the
    // dark (higher max) than humans; DSLR locks the auto loop;
    // linear is a debug pass-through.
    if (auto* v = C.FindCVar("r_eye_model")) {
        v->allowed_values = {
            "human", "cat", "owl",
            "dslr_iso100", "dslr_iso6400",
            "phone", "linear", "custom",
        };
        v->on_change = [](const pt::console::CVar& cv) {
            struct Preset {
                const char* name;
                float exp_min, exp_max, target, adapt;
                bool  auto_exposure;       // 0 forces manual r_exposure
                float manual_exp;
            };
            static const Preset presets[] = {
                // Human eye: comfortable adaptation range, ~1s adapt time.
                {"human",        0.05f,  4.0f,  0.18f, 0.20f, true,  1.5f},
                // Cats: rod-rich retina, ~6x dim-light sensitivity.
                {"cat",          0.05f, 12.0f,  0.18f, 0.30f, true,  1.5f},
                // Owls: nocturnal extreme, ~100x rod density of humans.
                {"owl",          0.05f, 30.0f,  0.18f, 0.35f, true,  1.5f},
                // DSLR locked at ISO 100: no auto, fixed exposure.
                {"dslr_iso100",  1.0f,   1.0f,  0.18f, 0.0f,  false, 1.0f},
                // DSLR locked at ISO 6400: 64x more gain.
                {"dslr_iso6400", 8.0f,   8.0f,  0.18f, 0.0f,  false, 8.0f},
                // Smartphone: auto, modest range.
                {"phone",        0.10f,  6.0f,  0.18f, 0.25f, true,  1.5f},
                // Linear: bypass exposure entirely (debug).
                {"linear",       1.0f,   1.0f,  0.18f, 0.0f,  false, 1.0f},
            };
            if (cv.value == "custom") return;
            auto& C2 = pt::console::Console::Get();
            for (const auto& p : presets) {
                if (cv.value == p.name) {
                    C2.SetCVarOverride("r_exposure_min",    std::to_string(p.exp_min));
                    C2.SetCVarOverride("r_exposure_max",    std::to_string(p.exp_max));
                    C2.SetCVarOverride("r_exposure_target", std::to_string(p.target));
                    C2.SetCVarOverride("r_eye_adapt_speed", std::to_string(p.adapt));
                    C2.SetCVarOverride("r_auto_exposure",   p.auto_exposure ? "1" : "0");
                    if (!p.auto_exposure) {
                        C2.SetCVarOverride("r_exposure", std::to_string(p.manual_exp));
                    }
                    return;
                }
            }
        };
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
    // Sky cvars: changing any of them invalidates accumulation.
    if (auto* v = C.FindCVar("r_sky_mode")) {
        v->allowed_values = {"gradient", "hdri", "procedural"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sun_elevation")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sun_azimuth")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_exposure")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sky_use_astronomical")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sky_hour")) {
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_sky_animate")) {
        v->allowed_values = {"0", "1"};
        // No accum reset on toggle: turning animation on/off doesn't
        // change the current hour; the next Tick handles continuity.
    }
    // r_sky_animate_rate has no validation/on_change wiring -- it's a
    // free-form float consumed by Tick(); changing it is felt the
    // next frame without any rebind.
    if (auto* v = C.FindCVar("r_show_stars")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_stars_mode")) {
        v->allowed_values = {"bsc", "procedural"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    if (auto* v = C.FindCVar("r_stars_twinkle")) {
        v->allowed_values = {"0", "1"};
        // Twinkle is per-frame in the shader; toggling doesn't need an
        // accum reset because the change is felt the next frame anyway.
    }
    if (auto* v = C.FindCVar("r_sky_lat") ) v->on_change = [this](const pt::console::CVar&){ accum_dirty_ = true; };
    if (auto* v = C.FindCVar("r_sky_lon") ) v->on_change = [this](const pt::console::CVar&){ accum_dirty_ = true; };
    if (auto* v = C.FindCVar("r_sky_time_offset")) v->on_change = [this](const pt::console::CVar&){ accum_dirty_ = true; };

    // City preset dropdown. Each entry is (name, lat, lon). Selecting
    // one writes lat/lon to the geographic cvars; "custom" leaves
    // them as-is so the user can dial in arbitrary coordinates.
    if (auto* v = C.FindCVar("r_sky_city")) {
        v->allowed_values = {
            "chennai", "mumbai", "delhi", "bangalore",
            "tokyo", "singapore", "beijing", "seoul",
            "london", "paris", "berlin", "moscow",
            "new_york", "san_francisco", "los_angeles", "chicago",
            "sao_paulo", "buenos_aires",
            "sydney", "auckland",
            "cairo", "cape_town",
            "reykjavik", "anchorage",
            "custom",
        };
        v->on_change = [this](const pt::console::CVar& cv) {
            // tz is the city's standard-time UTC offset in hours
            // (positive east). DST is intentionally ignored: this is
            // a sky simulator, not a clock app, and DST switches add
            // complexity that would obscure the astronomy.
            struct CityRow { const char* name; double lat; double lon; double tz; };
            static const CityRow cities[] = {
                {"chennai",        13.0827,   80.2707,   5.5},
                {"mumbai",         19.0760,   72.8777,   5.5},
                {"delhi",          28.6139,   77.2090,   5.5},
                {"bangalore",      12.9716,   77.5946,   5.5},
                {"tokyo",          35.6762,  139.6503,   9.0},
                {"singapore",       1.3521,  103.8198,   8.0},
                {"beijing",        39.9042,  116.4074,   8.0},
                {"seoul",          37.5665,  126.9780,   9.0},
                {"london",         51.5074,   -0.1278,   0.0},
                {"paris",          48.8566,    2.3522,   1.0},
                {"berlin",         52.5200,   13.4050,   1.0},
                {"moscow",         55.7558,   37.6173,   3.0},
                {"new_york",       40.7128,  -74.0060,  -5.0},
                {"san_francisco",  37.7749, -122.4194,  -8.0},
                {"los_angeles",    34.0522, -118.2437,  -8.0},
                {"chicago",        41.8781,  -87.6298,  -6.0},
                {"sao_paulo",     -23.5505,  -46.6333,  -3.0},
                {"buenos_aires",  -34.6037,  -58.3816,  -3.0},
                {"sydney",        -33.8688,  151.2093,  10.0},
                {"auckland",      -36.8485,  174.7633,  12.0},
                {"cairo",          30.0444,   31.2357,   2.0},
                {"cape_town",     -33.9249,   18.4241,   2.0},
                {"reykjavik",      64.1466,  -21.9426,   0.0},
                {"anchorage",      61.2181, -149.9003,  -9.0},
            };
            if (cv.value == "custom") { accum_dirty_ = true; return; }
            for (const auto& c : cities) {
                if (cv.value == c.name) {
                    auto& C2 = pt::console::Console::Get();
                    C2.SetCVarOverride("r_sky_lat", std::to_string(c.lat));
                    C2.SetCVarOverride("r_sky_lon", std::to_string(c.lon));
                    C2.SetCVarOverride("r_sky_tz_offset_hours", std::to_string(c.tz));
                    accum_dirty_ = true;
                    LOG_INFO("r_sky_city: '{}' -> lat={:.4f} lon={:.4f} tz={:+.1f}",
                             c.name, c.lat, c.lon, c.tz);
                    return;
                }
            }
            LOG_WARN("r_sky_city: '{}' has no preset; lat/lon unchanged",
                     std::string(cv.value));
        };
    }
    if (auto* v = C.FindCVar("r_sky_hour_local")) {
        v->allowed_values = {"0", "1"};
        v->on_change = [this](const pt::console::CVar&) { accum_dirty_ = true; };
    }
    // app_vsync / app_overlay_enabled / app_auto_open_console / dev_cheats:
    // boolean toggles -- accept 0|1.
    for (const char* n : {"app_vsync", "app_overlay_enabled",
                          "app_auto_open_console", "dev_cheats"}) {
        if (auto* v = C.FindCVar(n)) v->allowed_values = {"0", "1"};
    }

    // Slider ranges for numeric cvars. The web UI checks slider_max >
    // slider_min and renders a draggable range input + numeric readout
    // instead of a free-form text box. Pick ranges that cover the
    // useful design space, not the full numeric domain (e.g. r_spp
    // tops out at 32 even though the renderer accepts more) -- a
    // slider that needs a microscope to land on a sane value defeats
    // the purpose. Free-form input is still possible from the command
    // line for power users.
    auto set_slider = [&C](const char* name, float lo, float hi, float step) {
        if (auto* v = C.FindCVar(name)) {
            v->slider_min  = lo;
            v->slider_max  = hi;
            v->slider_step = step;
        }
    };
    set_slider("r_spp",             1.0f,   32.0f,  1.0f);
    set_slider("r_max_bounces",     1.0f,   16.0f,  1.0f);
    set_slider("r_exposure",        0.1f,    5.0f,  0.05f);
    set_slider("r_env_intensity",   0.0f,    5.0f,  0.05f);
    set_slider("r_sun_elevation", -90.0f,   90.0f,  0.5f);
    set_slider("r_sun_azimuth",     0.0f,  360.0f,  1.0f);
    set_slider("r_sky_hour",          0.0f,  24.0f,  0.01f);
    set_slider("r_sky_animate_rate",  0.0f,  24.0f,  0.05f);
    set_slider("r_sky_lat",         -90.0f,  90.0f,  0.1f);
    set_slider("r_sky_lon",      -180.0f,  180.0f,  0.1f);
    set_slider("cam_fov",          20.0f,  120.0f,  0.5f);
    set_slider("cam_speed",         0.1f,   30.0f,  0.1f);
    set_slider("cam_sprint_mult",   1.0f,   10.0f,  0.1f);
    set_slider("cam_sensitivity",   0.01f,   1.0f,  0.01f);
    set_slider("r_dof_aperture",        0.0f,   1.0f,  0.001f);
    set_slider("r_dof_focal_distance",  0.1f, 100.0f,  0.1f);
    set_slider("r_dof_blades",          0.0f,  16.0f,  1.0f);
    set_slider("r_bloom_threshold",     0.0f,  10.0f,  0.05f);
    set_slider("r_bloom_intensity",     0.0f,   1.0f,  0.005f);
    set_slider("r_bloom_mips",          1.0f,   5.0f,  1.0f);
    set_slider("r_bloom_radius",        0.5f,   3.0f,  0.05f);
    set_slider("r_lens_flare_intensity",      0.0f, 1.0f,  0.005f);
    set_slider("r_lens_flare_dispersion",     0.0f, 0.05f, 0.001f);
    set_slider("r_lens_flare_count",          1.0f, 6.0f,  1.0f);
    set_slider("r_lens_flare_threshold",      0.5f, 10.0f, 0.05f);

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

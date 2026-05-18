// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// pt_render_one_frame: thin harness CLI for the golden-image regression
// matrix (issue #45). Translates a stable test-facing flag set
//   --scene <path> --backend {software|metal|vulkan} --denoiser <kind>
//   --spp <N> --frames <N> --out <png-path> [--extra "<cvar> <value>; ..."]
// into the engine's existing smoke-test plumbing
//   demont --smoke-frames=N --r-backend=X --smoke-exec=<path> \
//          --smoke-capture-out=<png-path>
// and spawns the demont binary that sits alongside this tool in the
// build tree. ctest cells in tests/CMakeLists.txt invoke this wrapper
// instead of demont directly so the test surface stays stable even if
// the engine's CLI evolves (e.g. someone renames --smoke-frames= later;
// only this wrapper needs the fix-up). The wrapper itself never opens a
// window, never touches a GPU, and never links anything from the engine
// libs -- the actual rendering happens in the child demont process via
// the smoke-test loop wired in src/engine/Engine.cpp.
//
// Design choice: the wrapper writes a small "synthesised" pre-exec cfg
// to a temp file that injects --denoiser / --spp / --extra into a single
// script + chains the operator's --scene cfg in via the `exec` command.
// That keeps the engine side simple (one pt_smoke_exec entry point) and
// lets the wrapper add per-cell knobs without touching the engine again
// every time the matrix grows. See the comments around BuildSmokeExec
// for the file-format contract.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr int kExitOk          = 0;
constexpr int kExitUsageError  = 2;
constexpr int kExitSpawnError  = 3;
constexpr int kExitIoError     = 4;

void PrintUsage(std::FILE* out) {
    std::fprintf(out,
        "Usage: pt_render_one_frame --scene <cfg> --backend X --out <png> [options]\n"
        "\n"
        "Headless render harness for the golden-image regression matrix\n"
        "(issue #45). Spawns the demont binary sitting next to this tool\n"
        "with the engine's smoke-test flags wired up so it renders N\n"
        "frames, captures the final frame to PNG, and exits cleanly.\n"
        "\n"
        "Required:\n"
        "  --scene PATH         Path to a console-script .cfg fixture\n"
        "                       (loaded by the engine before backend init).\n"
        "  --backend NAME       One of software | metal | vulkan.\n"
        "  --out PATH           Destination PNG for the final frame.\n"
        "\n"
        "Optional:\n"
        "  --denoiser KIND      Sets r_denoiser (default: off).\n"
        "  --spp N              Sets r_spp (default: leave fixture value).\n"
        "  --frames N           Engine smoke-test frame budget (default 32).\n"
        "  --extra CMDS         Semicolon-separated console-script commands\n"
        "                       inlined into the synthesised pre-exec cfg.\n"
        "  --demont PATH        Override path to the demont binary (default:\n"
        "                       same directory as this tool, OS-suffixed).\n"
        "  -h, --help           Show this help.\n"
        "\n"
        "Exit codes:\n"
        "  0 = demont exited 0 (smoke-test passed, PNG written)\n"
        "  2 = bad arguments to this wrapper\n"
        "  3 = couldn't spawn the demont child / child returned non-zero\n"
        "  4 = couldn't write the synthesised smoke-exec cfg\n");
}

struct Args {
    std::string scenePath;
    std::string backend;
    std::string outPath;
    std::string denoiser    = "off";
    std::string spp;          // empty => leave fixture's value
    int         frames       = 32;
    std::string extra;
    std::string demontPath;   // empty => derive from argv[0]
    bool        helpRequested = false;
};

bool ParsePositiveInt(std::string_view s, int& out) {
    if (s.empty()) return false;
    constexpr int kCap = 1'000'000;   // sanity cap; well above any real --frames/--spp
    int n = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        const int digit = c - '0';
        // Check BEFORE the multiply/add so a long numeric like
        // "999999999999999999999" can't overflow `int` and invoke UB
        // in the validator. The cap is 1e6, so n*10 stays well under
        // INT_MAX as long as n is already below the cap.
        if (n > (kCap - digit) / 10) return false;
        n = n * 10 + digit;
    }
    if (n <= 0) return false;
    out = n;
    return true;
}

bool ParseArgs(int argc, char** argv, Args& a) {
    auto takeValue = [&](int& i, std::string_view flag, std::string& dst) -> bool {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "pt_render_one_frame: %.*s requires a value\n",
                         int(flag.size()), flag.data());
            return false;
        }
        ++i;
        dst = argv[i];
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string_view s = argv[i];
        if (s == "-h" || s == "--help") { a.helpRequested = true; return true; }
        else if (s == "--scene")        { if (!takeValue(i, s, a.scenePath))    return false; }
        else if (s == "--backend")      { if (!takeValue(i, s, a.backend))      return false; }
        else if (s == "--out")          { if (!takeValue(i, s, a.outPath))      return false; }
        else if (s == "--denoiser")     { if (!takeValue(i, s, a.denoiser))     return false; }
        else if (s == "--spp") {
            std::string v;
            if (!takeValue(i, s, v)) return false;
            // Same positive-int validation as --frames. r_spp has no
            // allowed-values gate engine-side and CVar::GetInt() falls
            // back to the default on parse failure, so an unvalidated
            // `--spp abc` would silently render at the default sample
            // count instead of failing here.
            int spp_n = 0;
            if (!ParsePositiveInt(v, spp_n)) {
                std::fprintf(stderr,
                    "pt_render_one_frame: --spp: invalid positive integer '%s'\n",
                    v.c_str());
                return false;
            }
            a.spp = v;
        }
        else if (s == "--extra")        { if (!takeValue(i, s, a.extra))        return false; }
        else if (s == "--demont")       { if (!takeValue(i, s, a.demontPath))   return false; }
        else if (s == "--frames") {
            std::string v;
            if (!takeValue(i, s, v)) return false;
            if (!ParsePositiveInt(v, a.frames)) {
                std::fprintf(stderr,
                    "pt_render_one_frame: --frames: invalid integer '%s'\n", v.c_str());
                return false;
            }
        }
        else {
            std::fprintf(stderr,
                "pt_render_one_frame: unknown flag '%s'\n", argv[i]);
            return false;
        }
    }
    if (a.helpRequested) return true;
    if (a.scenePath.empty()) {
        std::fprintf(stderr, "pt_render_one_frame: --scene is required\n");
        return false;
    }
    if (a.backend.empty()) {
        std::fprintf(stderr, "pt_render_one_frame: --backend is required\n");
        return false;
    }
    if (a.outPath.empty()) {
        std::fprintf(stderr, "pt_render_one_frame: --out is required\n");
        return false;
    }
    if (a.backend != "software" && a.backend != "metal" && a.backend != "vulkan") {
        std::fprintf(stderr,
            "pt_render_one_frame: --backend must be one of "
            "{software,metal,vulkan}; got '%s'\n", a.backend.c_str());
        return false;
    }
    // Validate --denoiser against the full superset of r_denoiser
    // values the engine accepts (per the PT_CVAR description in
    // src/engine/Engine.cpp). The engine does NOT have an
    // allowed-values gate on r_denoiser; an unrecognised string is
    // silently treated as `off`. Without this check a typo like
    // `--denoiser svgf_basc` would render at the noisier `off` setting
    // and -- worse, at regen time -- get committed as an "off"
    // baseline misfiled under the misspelled name. The full superset
    // is validated here regardless of backend; the engine handles
    // platform availability at runtime (e.g. metalfx on Vulkan reverts
    // to off).
    {
        static constexpr std::string_view kDenoiserAllowed[] = {
            "off",
            "metalfx",
            "svgf_basic", "svgf_atrous",
            "svgf_basic_metalfx", "svgf_atrous_metalfx",
            "nrd",
            "optix_hdr", "optix_hdr_aov",
            "optix_temporal_hdr", "optix_temporal_hdr_aov",
        };
        bool ok = false;
        for (auto v : kDenoiserAllowed) {
            if (a.denoiser == v) { ok = true; break; }
        }
        if (!ok) {
            std::fprintf(stderr,
                "pt_render_one_frame: --denoiser '%s' is not a recognised "
                "r_denoiser value. Supported: off, metalfx, svgf_basic, "
                "svgf_atrous, svgf_basic_metalfx, svgf_atrous_metalfx, "
                "nrd, optix_hdr, optix_hdr_aov, optix_temporal_hdr, "
                "optix_temporal_hdr_aov.\n",
                a.denoiser.c_str());
            return false;
        }
    }
    // Validate scene path is a *readable regular file* in the wrapper.
    // A bare existence check would pass a directory (which fopen()
    // happily opens on POSIX, then fread returns 0 and the engine's
    // smoke-exec sees an empty script -> renders the default scene ->
    // pixel diff failure later). is_regular_file rules out
    // directories / symlinks-to-directories / device nodes; a probe
    // open rules out permission-denied files we couldn't read anyway.
    {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(a.scenePath, ec) || ec) {
            std::fprintf(stderr,
                "pt_render_one_frame: --scene '%s' is not a regular file%s%s\n",
                a.scenePath.c_str(),
                ec ? ": " : "",
                ec ? ec.message().c_str() : "");
            return false;
        }
        std::FILE* probe = std::fopen(a.scenePath.c_str(), "rb");
        if (probe == nullptr) {
            std::fprintf(stderr,
                "pt_render_one_frame: --scene '%s' is not readable: %s\n",
                a.scenePath.c_str(), std::strerror(errno));
            return false;
        }
        std::fclose(probe);
    }
    return true;
}

// Locate the demont binary that lives alongside this tool. Build trees
// produce side-by-side targets under build/<preset>/tools/pt_render_one_frame/
// and build/<preset>/src/app/, so the lookup walks up + over from argv[0].
// Operator can short-circuit the search via --demont PATH.
std::filesystem::path ResolveDemontPath(const std::filesystem::path& self_arg0,
                                        const std::string& override_path) {
    namespace fs = std::filesystem;
    if (!override_path.empty()) return fs::path(override_path);

    std::error_code ec;
    fs::path self = fs::weakly_canonical(self_arg0, ec);
    if (ec) self = self_arg0;
    const fs::path self_dir = self.parent_path();

    // Suffix is .exe on Windows, none elsewhere. Build-tree layout
    // assumed: <build>/tools/pt_render_one_frame/pt_render_one_frame[.exe]
    // and        <build>/src/app/demont[.exe]
#if defined(_WIN32)
    const char* exe_suffix = ".exe";
#else
    const char* exe_suffix = "";
#endif
    const std::string demont_name = std::string("demont") + exe_suffix;

    // Candidate 1: <build>/src/app/demont (typical CMake layout).
    fs::path build_root = self_dir;
    if (!build_root.empty()) build_root = build_root.parent_path();   // tools/
    if (!build_root.empty()) build_root = build_root.parent_path();   // <build>/
    if (!build_root.empty()) {
        fs::path c1 = build_root / "src" / "app" / demont_name;
        if (fs::exists(c1, ec)) return c1;
    }
    // Candidate 2: sibling in same dir (single-config / dev layouts).
    fs::path c2 = self_dir / demont_name;
    if (fs::exists(c2, ec)) return c2;
    // Candidate 3: relative ../../src/app/demont from self.
    fs::path c3 = self_dir / ".." / ".." / "src" / "app" / demont_name;
    if (fs::exists(c3, ec)) return fs::weakly_canonical(c3, ec);

    // Last resort: emit the most-likely-correct candidate and let the
    // spawn fail loudly if it's wrong. Operator can pass --demont to
    // pin the path explicitly.
    return build_root.empty() ? c2 : (build_root / "src" / "app" / demont_name);
}

// Walk up from the wrapper exe looking for an `assets/` subdirectory.
// Found dir is the asset root we hand to demont via DEMONT_ASSET_ROOT;
// the engine's pt::ResolveAssetPath() consumes it in lieu of a CWD-
// relative open. Used to cover ctest's per-cell golden_workdir/<cell>/
// CWD where the engine would otherwise miss assets/hdri/*.hdr +
// assets/stars/BSC5.dat. Falls back to empty path -> caller skips the
// env-var injection -> engine resolves via exe-grandparent (packaged-
// install layout).
std::filesystem::path ResolveAssetRoot(const std::filesystem::path& self_arg0) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path self = fs::weakly_canonical(self_arg0, ec);
    if (ec) self = self_arg0;
    fs::path dir = self.parent_path();
    // 6 levels up covers <repo>/build/<preset>/tools/pt_render_one_frame/
    // and the packaged <install>/bin/ layout, with margin.
    for (int i = 0; i < 6 && !dir.empty(); ++i) {
        if (fs::is_directory(dir / "assets", ec)) return dir;
        fs::path parent = dir.parent_path();
        if (parent == dir) break;   // hit filesystem root
        dir = parent;
    }
    return {};
}

// Build a synthesised console-script that the engine exec's via
// --smoke-exec=<path>. Format is one console command per line; blank
// lines and lines starting with `#` are ignored by the parser. The
// file:
//   1. Inlines the operator's --scene fixture verbatim (read from
//      disk here, NOT chained through the console `exec` command --
//      `exec` only logs nested ExecuteScript failures and returns ok
//      to its caller, so a typo in the fixture would be silently
//      ignored by the engine's strict smoke-exec path).
//   2. Sets r_denoiser to the operator's pick.
//   3. Optionally sets r_spp.
//   4. Inlines --extra "<cmds>" verbatim (caller is responsible for
//      keeping quoting sane on the host shell side).
// Returns the path on success (empty filesystem::path on I/O failure).
std::filesystem::path BuildSmokeExec(const Args& a) {
    namespace fs = std::filesystem;

    // Slurp the scene fixture. ParseArgs already validated it as a
    // readable regular file, so a failure here is an I/O race
    // (deleted between validate + read) -- treat as fatal.
    std::FILE* sf = std::fopen(a.scenePath.c_str(), "rb");
    if (sf == nullptr) {
        std::fprintf(stderr,
            "pt_render_one_frame: failed to reopen --scene '%s': %s\n",
            a.scenePath.c_str(), std::strerror(errno));
        return {};
    }
    std::string scene_body;
    char rbuf[4096];
    while (auto rn = std::fread(rbuf, 1, sizeof(rbuf), sf)) {
        scene_body.append(rbuf, rn);
    }
    const bool scene_read_err = (std::ferror(sf) != 0);
    std::fclose(sf);
    if (scene_read_err) {
        std::fprintf(stderr,
            "pt_render_one_frame: read error on --scene '%s'\n",
            a.scenePath.c_str());
        return {};
    }

    // The no-arg `temp_directory_path()` throws on failure. This TU
    // builds under the project's `-fno-exceptions` / `/EHs-c-` policy,
    // so a throw here would terminate the wrapper instead of returning
    // the documented kExitIoError. Use the error_code overload.
    std::error_code tmp_ec;
    fs::path tmp_dir = fs::temp_directory_path(tmp_ec);
    if (tmp_ec) {
        std::fprintf(stderr,
            "pt_render_one_frame: temp_directory_path failed: %s\n",
            tmp_ec.message().c_str());
        return {};
    }
    // Randomised name so parallel ctest runs don't clobber each other's
    // temp script. ctest's job control runs cells in parallel by default.
    std::random_device rd;
    std::mt19937_64 gen(rd());
    const auto rnd = gen();
    char buf[40];
    std::snprintf(buf, sizeof(buf), "pt_smoke_%016llx.cfg",
                  static_cast<unsigned long long>(rnd));
    fs::path out = tmp_dir / buf;

    std::ofstream f(out, std::ios::binary);
    if (!f.is_open()) return {};

    f << "# pt_render_one_frame: synthesised --smoke-exec cfg\n";
    f << "# Generated for backend=" << a.backend
      << " denoiser=" << a.denoiser
      << " spp=" << (a.spp.empty() ? "<fixture>" : a.spp)
      << " frames=" << a.frames << "\n";
    // Scene fixture loads FIRST (inlined verbatim, not via `exec`) so
    // the wrapper's per-cell overrides below can override anything the
    // fixture sets. cornell_csg.cfg pins r_spp 1, so if the wrapper
    // emitted `r_spp` before the fixture the fixture would silently
    // overwrite the operator's `--spp` value. The "more-specific wins"
    // overlay here is wrapper flag > fixture > engine default.
    f << "# --- inlined from " << a.scenePath << " ---\n";
    f << scene_body;
    if (!scene_body.empty() && scene_body.back() != '\n') f << '\n';
    f << "# --- end inlined fixture ---\n";
    f << "r_denoiser " << a.denoiser << "\n";
    if (!a.spp.empty()) {
        f << "r_spp " << a.spp << "\n";
    }
    if (!a.extra.empty()) {
        // Pass-through. Console::ExecuteScript already splits on `\n` and
        // `;` internally, so the operator can stuff multiple cmds into
        // --extra without us re-parsing here. Caller is responsible for
        // shell-quoting the value sensibly on its end.
        f << a.extra << "\n";
    }
    if (!f) return {};
    return out;
}

#if defined(_WIN32)
// CreateProcess takes a single command line string -- argv arrays don't
// exist on the Win32 native API. Build a properly-quoted command line
// per the MSVC argument-parsing rules
// (https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args).
std::string QuoteWin32Arg(const std::string& a) {
    if (a.find_first_of(" \t\"") == std::string::npos && !a.empty()) {
        return a;  // no quoting needed
    }
    std::string out;
    out.push_back('"');
    for (size_t i = 0; i < a.size(); ++i) {
        size_t backslashes = 0;
        while (i < a.size() && a[i] == '\\') { ++backslashes; ++i; }
        if (i == a.size()) {
            out.append(backslashes * 2, '\\');
            break;
        } else if (a[i] == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
        } else {
            out.append(backslashes, '\\');
            out.push_back(a[i]);
        }
    }
    out.push_back('"');
    return out;
}

int SpawnAndWait(const std::filesystem::path& exe,
                 const std::vector<std::string>& args) {
    std::string cmdline = QuoteWin32Arg(exe.string());
    for (const auto& a : args) {
        cmdline.push_back(' ');
        cmdline.append(QuoteWin32Arg(a));
    }
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<char> mutable_cmdline(cmdline.begin(), cmdline.end());
    mutable_cmdline.push_back('\0');
    if (!CreateProcessA(exe.string().c_str(),
                        mutable_cmdline.data(),
                        nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &si, &pi)) {
        std::fprintf(stderr,
            "pt_render_one_frame: CreateProcessA failed for '%s' (err=%lu)\n",
            exe.string().c_str(),
            static_cast<unsigned long>(GetLastError()));
        return kExitSpawnError;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exit_code);
}
#else
int SpawnAndWait(const std::filesystem::path& exe,
                 const std::vector<std::string>& args) {
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr,
            "pt_render_one_frame: fork failed: %s\n", std::strerror(errno));
        return kExitSpawnError;
    }
    if (pid == 0) {
        // Child: build argv. execvp expects char* (mutable strings) per
        // POSIX, but the strings are not actually mutated by the kernel
        // before exec replaces the address space; the const_cast is the
        // standard idiom.
        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(exe.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execv(exe.c_str(), argv.data());
        // execv only returns on failure.
        std::fprintf(stderr,
            "pt_render_one_frame: execv '%s' failed: %s\n",
            exe.c_str(), std::strerror(errno));
        _exit(127);
    }
    // Parent.
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        std::fprintf(stderr,
            "pt_render_one_frame: waitpid failed: %s\n", std::strerror(errno));
        return kExitSpawnError;
    }
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) {
        std::fprintf(stderr,
            "pt_render_one_frame: demont died on signal %d\n",
            WTERMSIG(status));
        return kExitSpawnError;
    }
    return kExitSpawnError;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!ParseArgs(argc, argv, a)) {
        PrintUsage(stderr);
        return kExitUsageError;
    }
    if (a.helpRequested) {
        PrintUsage(stdout);
        return kExitOk;
    }

    // Hermetic render setup: demont reads `demont.cfg` and
    // `autoexec.cfg` from its CWD before applying the smoke-exec
    // fixture, and the fixture doesn't necessarily pin every
    // archived cvar (DOF, volumetrics, etc. may not be in the
    // fixture's pin list). Clean those files from the wrapper's
    // CWD so the render starts from engine defaults + fixture, not
    // engine defaults + last run's archive + fixture. ctest sets
    // CWD to the per-cell workdir (tests/CMakeLists.txt) so this
    // touch is bounded to that cell's directory.
    {
        std::error_code rm_ec;
        std::filesystem::remove("demont.cfg",   rm_ec);
        std::filesystem::remove("autoexec.cfg", rm_ec);
    }

    const std::filesystem::path smoke_exec = BuildSmokeExec(a);
    if (smoke_exec.empty()) {
        std::fprintf(stderr,
            "pt_render_one_frame: could not write synthesised smoke-exec cfg\n");
        return kExitIoError;
    }

    const std::filesystem::path demont =
        ResolveDemontPath(std::filesystem::path(argv[0]), a.demontPath);

    // Propagate the asset root to the child engine via DEMONT_ASSET_ROOT.
    // ctest sets each cell's CWD to golden_workdir/<cell>/, which makes
    // the engine's CWD-relative asset opens (env_map sunset.hdr, BSC5)
    // miss the actual files under <repo>/assets/. Resolving the root
    // here from the wrapper's exe location (which is fixed by the build
    // tree layout) and forwarding via env var is robust to whatever CWD
    // ctest chose. The child inherits the parent env on both Win32
    // (CreateProcessA with lpEnvironment=nullptr) and POSIX (fork+exec).
    {
        const std::filesystem::path asset_root =
            ResolveAssetRoot(std::filesystem::path(argv[0]));
        if (!asset_root.empty()) {
            const std::string s = asset_root.string();
#if defined(_WIN32)
            _putenv_s("DEMONT_ASSET_ROOT", s.c_str());
#else
            setenv("DEMONT_ASSET_ROOT", s.c_str(), 1);
#endif
        }
    }

    // error_code overload (no-throw): the no-arg `exists()` throws on
    // a filesystem error, but this TU is -fno-exceptions. Use the
    // explicit ec overload so a probe failure surfaces as
    // kExitSpawnError instead of terminating.
    std::error_code demont_ec;
    const bool demont_present = std::filesystem::exists(demont, demont_ec);
    if (demont_ec || !demont_present) {
        const std::string demont_err_msg = demont_ec ? demont_ec.message() : std::string();
        std::fprintf(stderr,
            "pt_render_one_frame: demont binary not found at '%s'%s%s%s. "
            "Pass --demont PATH to override.\n",
            demont.string().c_str(),
            demont_ec ? " (" : "",
            demont_ec ? demont_err_msg.c_str() : "",
            demont_ec ? ")" : "");
        std::error_code ec;
        std::filesystem::remove(smoke_exec, ec);
        return kExitSpawnError;
    }

    std::vector<std::string> child_args = {
        std::string("--smoke-frames=") + std::to_string(a.frames),
        std::string("--r-backend=") + a.backend,
        std::string("--smoke-exec=") + smoke_exec.string(),
        std::string("--smoke-capture-out=") + a.outPath,
    };

    const int rc = SpawnAndWait(demont, child_args);

    // Best-effort cleanup of the synthesised cfg. We don't fail the
    // wrapper on cleanup error -- the failure mode is "leftover temp
    // file the OS will GC eventually".
    std::error_code ec;
    std::filesystem::remove(smoke_exec, ec);

    if (rc != 0) {
        std::fprintf(stderr,
            "pt_render_one_frame: demont exited with code %d\n", rc);
        // The usage text documents exit 3 (kExitSpawnError) for "child
        // returned non-zero". Honour that contract regardless of the
        // child's specific exit code; callers (ctest cells, scripts)
        // gate on this wrapper's exit, not demont's.
        return kExitSpawnError;
    }
    return kExitOk;
}

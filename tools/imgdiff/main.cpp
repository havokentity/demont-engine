// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
//
// imgdiff: PNG-to-PNG comparison CLI. Groundwork for the golden-image
// regression matrix (#45). Exits 0 if the candidate matches the
// reference within the supplied thresholds; non-zero otherwise.
//
// Usage:
//   imgdiff actual.png golden.png [--max-delta N] [--mean-delta M]
//                                  [--fail-percent P] [--diff out.png]
//
// Threshold semantics live in ImgDiff.h. All three numeric flags
// default to 0 -> byte-identical match required. Loosen as needed for
// AA / dithering tolerance in golden goldens.
//
// PNG I/O via vendored stb_image / stb_image_write headers. Output (in
// addition to the exit code) is one line of "key=value ..." stats to
// stdout so it's easy to grep + parse from CI logs.

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "ImgDiff.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr int kExitOk           = 0;
constexpr int kExitMismatch     = 1;  // diff above thresholds
constexpr int kExitUsageError   = 2;  // bad args / unknown flag
constexpr int kExitLoadError    = 3;  // PNG load failed
constexpr int kExitSizeMismatch = 4;  // images differ in dimensions

void PrintUsage(std::FILE* out) {
    std::fprintf(out,
        "Usage: imgdiff actual.png golden.png [options]\n"
        "\n"
        "Compare two RGBA images and exit non-zero if the per-pixel L2\n"
        "distance exceeds the supplied thresholds. Used by the golden-\n"
        "image regression matrix.\n"
        "\n"
        "Options:\n"
        "  --max-delta N       Per-pixel L2 above N counts as a 'bad' pixel.\n"
        "                      Default 0 (any non-zero diff is bad).\n"
        "  --mean-delta M      Fail if mean per-pixel L2 > M. Default 0.\n"
        "  --fail-percent P    Fail if percentage of bad pixels > P. Default 0.\n"
        "  --diff path.png     Write a colorized heatmap PNG of per-pixel deltas.\n"
        "  -h, --help          Show this help.\n"
        "\n"
        "Exit codes:\n"
        "  0 = pass (within thresholds)\n"
        "  1 = mismatch (above thresholds)\n"
        "  2 = usage error\n"
        "  3 = failed to load one of the input PNGs\n"
        "  4 = inputs differ in dimensions\n");
}

bool ParseDouble(std::string_view s, double& out) {
    // std::from_chars on double is C++17 in the spec but only widely
    // shipped in libstdc++ 11+ / libc++ 17+ / MSVC STL 19.21+. All
    // toolchains this project supports (MSVC 2022, AppleClang 15+,
    // GCC 13+) ship it. Strict parse: rejects "1.0junk" by checking
    // that the whole string-view was consumed.
    const char* first = s.data();
    const char* last  = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec != std::errc{}) return false;
    if (ptr != last)        return false;
    return true;
}

struct CliArgs {
    const char* actualPath = nullptr;
    const char* goldenPath = nullptr;
    const char* diffPath   = nullptr;
    pt::imgdiff::Thresholds thresholds;
    bool helpRequested     = false;
};

// Parse argv into CliArgs. Returns true on success, false on usage
// error (in which case an error message has already been printed to
// stderr).
bool ParseArgs(int argc, char** argv, CliArgs& out) {
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];

        if (a == "-h" || a == "--help") {
            out.helpRequested = true;
            return true;
        }

        // All value-taking flags need a follow-on argv.
        auto takeValue = [&](std::string_view flag, double& dst) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "imgdiff: %.*s requires a value\n",
                             int(flag.size()), flag.data());
                return false;
            }
            ++i;
            if (!ParseDouble(argv[i], dst)) {
                std::fprintf(stderr, "imgdiff: %.*s: invalid number '%s'\n",
                             int(flag.size()), flag.data(), argv[i]);
                return false;
            }
            return true;
        };

        if (a == "--max-delta") {
            if (!takeValue(a, out.thresholds.maxDelta)) return false;
        } else if (a == "--mean-delta") {
            if (!takeValue(a, out.thresholds.meanDelta)) return false;
        } else if (a == "--fail-percent") {
            if (!takeValue(a, out.thresholds.failPercent)) return false;
        } else if (a == "--diff") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "imgdiff: --diff requires a path\n");
                return false;
            }
            ++i;
            out.diffPath = argv[i];
        } else if (!a.empty() && a.front() == '-') {
            std::fprintf(stderr, "imgdiff: unknown flag '%s'\n", argv[i]);
            return false;
        } else {
            if (positional == 0)      out.actualPath = argv[i];
            else if (positional == 1) out.goldenPath = argv[i];
            else {
                std::fprintf(stderr, "imgdiff: unexpected extra argument '%s'\n",
                             argv[i]);
                return false;
            }
            ++positional;
        }
    }

    if (out.helpRequested) return true;

    if (!out.actualPath || !out.goldenPath) {
        std::fprintf(stderr, "imgdiff: expected two positional PNG paths\n");
        return false;
    }
    return true;
}

// Wrap stb_image's malloc-returning load into a vector for RAII. Forces
// 4-channel RGBA8 output regardless of source channel count -- matches
// the ImgDiff::Compute contract.
struct LoadedImage {
    std::vector<uint8_t> pixels;
    int width  = 0;
    int height = 0;
};

bool LoadRgba8(const char* path, LoadedImage& out) {
    int w = 0, h = 0, ch = 0;
    // desired_channels=4 -> always RGBA8; stb fills the alpha with 255
    // for sources without alpha. Source-channel-count goes into `ch`
    // for diagnostics but we never read it again after this.
    uint8_t* data = stbi_load(path, &w, &h, &ch, 4);
    if (!data) {
        std::fprintf(stderr, "imgdiff: failed to load '%s': %s\n",
                     path, stbi_failure_reason());
        return false;
    }
    const size_t bytes = size_t(w) * size_t(h) * 4u;
    out.pixels.assign(data, data + bytes);
    out.width  = w;
    out.height = h;
    stbi_image_free(data);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage(stderr);
        return kExitUsageError;
    }
    if (args.helpRequested) {
        PrintUsage(stdout);
        return kExitOk;
    }

    LoadedImage actual;
    LoadedImage golden;
    if (!LoadRgba8(args.actualPath, actual)) return kExitLoadError;
    if (!LoadRgba8(args.goldenPath, golden)) return kExitLoadError;

    if (actual.width != golden.width || actual.height != golden.height) {
        std::fprintf(stderr,
            "imgdiff: dimension mismatch: actual=%dx%d golden=%dx%d\n",
            actual.width, actual.height, golden.width, golden.height);
        return kExitSizeMismatch;
    }

    std::vector<uint8_t> diffPixels;
    uint8_t* diffPtr = nullptr;
    if (args.diffPath) {
        diffPixels.resize(size_t(actual.width) * size_t(actual.height) * 4u);
        diffPtr = diffPixels.data();
    }

    const pt::imgdiff::DiffStats stats = pt::imgdiff::Compute(
        actual.pixels.data(),
        golden.pixels.data(),
        uint32_t(actual.width),
        uint32_t(actual.height),
        args.thresholds.maxDelta,
        diffPtr);

    const bool ok = pt::imgdiff::Passes(stats, args.thresholds);

    // One-line key=value stats so CI logs can grep them out. `result` is
    // the human-friendly verdict; the exit code is the machine signal.
    std::printf(
        "result=%s width=%d height=%d total=%llu bad=%llu "
        "max_delta=%.4f mean_delta=%.6f rms_delta=%.6f bad_percent=%.6f "
        "thr_max=%.4f thr_mean=%.6f thr_fail_percent=%.6f\n",
        ok ? "pass" : "fail",
        actual.width, actual.height,
        (unsigned long long)stats.totalPixels,
        (unsigned long long)stats.badPixels,
        stats.maxDelta, stats.meanDelta, stats.rmsDelta, stats.badPercent,
        args.thresholds.maxDelta, args.thresholds.meanDelta,
        args.thresholds.failPercent);

    if (args.diffPath) {
        const int stride = actual.width * 4;
        if (!stbi_write_png(args.diffPath, actual.width, actual.height,
                            4, diffPixels.data(), stride)) {
            std::fprintf(stderr, "imgdiff: failed to write diff PNG '%s'\n",
                         args.diffPath);
            // Reported separately from the comparison verdict: the diff
            // PNG is a diagnostic artefact, and a failure to write it
            // shouldn't flip a passing comparison to a fail.
            return ok ? kExitLoadError : kExitMismatch;
        }
    }

    return ok ? kExitOk : kExitMismatch;
}

#include "core/Jobs/JobSystem.h"
#include "core/Log.h"
#include "core/Memory/Memory.h"
#include "engine/Engine.h"

#include <fmt/format.h>
#include <unistd.h>

namespace {

// ASCII variant of the hex + ray-bounce glyph. Hex frame in cyan,
// letters "D · M · T" (the acronym) at the bounce points in cyan,
// magenta dots for hit-points, "PT" anchor for PathTracer at the
// bottom inside the hex.
void PrintBootLogo() {
    const bool tty  = ::isatty(fileno(stderr)) != 0;
    const char* CY  = tty ? "\033[38;2;0;240;255m"   : "";  // electric cyan
    const char* MG  = tty ? "\033[38;2;255;58;140m"  : "";  // magenta
    const char* DM  = tty ? "\033[2;38;2;111;120;146m" : "";// dim grey
    const char* WT  = tty ? "\033[1;38;2;231;234;242m" : "";// bright fg
    const char* RS  = tty ? "\033[0m"                : "";

    fmt::print(stderr, "\n");
    fmt::print(stderr, "    {CY}__________{RS}\n",
               fmt::arg("CY", CY), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CY}/{RS}          {CY}\\{RS}\n",
               fmt::arg("CY", CY), fmt::arg("RS", RS));
    fmt::print(stderr, "  {CY}/{RS}  {WT}D{RS} {MG}.{RS} {WT}M{RS} {MG}.{RS} {WT}T{RS}  {CY}\\{RS}\n",
               fmt::arg("CY", CY), fmt::arg("MG", MG),
               fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "  {CY}\\{RS}    {MG}\\{RS}  {MG}/{RS}     {CY}/{RS}\n",
               fmt::arg("CY", CY), fmt::arg("MG", MG), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CY}\\{RS}    {WT}P T{RS}    {CY}/{RS}\n",
               fmt::arg("CY", CY), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "    {CY}\\__________/{RS}\n",
               fmt::arg("CY", CY), fmt::arg("RS", RS));
    fmt::print(stderr, "\n");
    fmt::print(stderr, "  {WT}DeMonT{RS} {CY}PathTracer{RS}   {DM}v0.1.0{RS}\n",
               fmt::arg("WT", WT), fmt::arg("CY", CY),
               fmt::arg("DM", DM), fmt::arg("RS", RS));
    fmt::print(stderr, "  {DM}De Monte Carlo-esque Tracer{RS}\n",
               fmt::arg("DM", DM), fmt::arg("RS", RS));
    fmt::print(stderr, "\n");
}

}  // namespace

int main() {
    PrintBootLogo();
    pt::engine::Engine engine;
    if (!engine.Init()) {
        LOG_ERROR("Engine init failed");
        return 1;
    }

    // Smoke-test the job system (Phase 1 acceptance).
    if (auto* js = pt::jobs::JobSystem::Instance(); js != nullptr) {
        bool ran = false;
        js->Run([&ran]{ ran = true; });
        LOG_INFO("JobSystem smoke test: ran = {}", ran);
    }

    engine.Run();

    pt::mem::PrintReport();
    return 0;
}

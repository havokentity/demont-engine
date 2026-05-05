#include "core/Jobs/JobSystem.h"
#include "core/Log.h"
#include "core/Memory/Memory.h"
#include "engine/Engine.h"

#include <fmt/format.h>
#include <unistd.h>

namespace {

// 13-line trippy hex with internal scaffolding. Same shape that
// pt::engine::Engine::EmitWelcomeBanner pushes through LOG_INFO so it
// shows up in the in-window overlay and the web console too -- this
// version just adds 24-bit ANSI colour for terminals.
void PrintBootLogo() {
    const bool tty  = ::isatty(fileno(stderr)) != 0;
    const char* CY  = tty ? "\033[38;2;0;240;255m"     : "";  // electric cyan
    const char* CD  = tty ? "\033[38;2;0;130;160m"     : "";  // dim cyan
    const char* MG  = tty ? "\033[38;2;255;58;140m"    : "";  // magenta
    const char* MB  = tty ? "\033[1;38;2;255;120;200m" : "";  // bright magenta
    const char* WT  = tty ? "\033[1;38;2;231;234;242m" : "";  // bright fg
    const char* DM  = tty ? "\033[2;38;2;111;120;146m" : "";  // dim grey
    const char* RS  = tty ? "\033[0m"                  : "";

    // 13 lines: hex frame (cyan, shaded ░▒▓█), DMT triangle inscribed
    // inside (pink edges + bounces), letters (bright white).
    fmt::print(stderr, "\n");
    fmt::print(stderr, "        {CD}░▒▓{CY}██████████{CD}▓▒░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("RS", RS));
    fmt::print(stderr, "     {CD}░▒▓{CY}██████████████{CD}▓▒░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}░▓{CY}██░{RS}         {WT}D{RS}         {CD}░{CY}██{CD}▓░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD),
               fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}▒{CY}█{CD}░{RS}          {MG}╱ ╲{RS}          {CD}░{CY}█{CD}▒{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD),
               fmt::arg("MG", MG), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}▓{CY}█{CD}░{RS}         {MG}╱ {MB}◉{MG} ╲{RS}          {CD}░{CY}█{CD}▓{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD),
               fmt::arg("MG", MG), fmt::arg("MB", MB), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CY}█{CD}░{RS}         {MG}╱ {WT}─{MB}•{WT}─{MG} ╲{RS}          {CD}░{CY}█{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("MG", MG),
               fmt::arg("MB", MB), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}▓{CY}█{CD}░{RS}       {MG}╱{RS}   {MB}◉{WT}│{MB}◉{RS}  {MG}╲{RS}        {CD}░{CY}█{CD}▓{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("MG", MG),
               fmt::arg("MB", MB), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CY}█{CD}░{RS}       {MG}╱  ─{MB}◉{WT} • {MB}◉{MG}─  ╲{RS}       {CD}░{CY}█{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("MG", MG),
               fmt::arg("MB", MB), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}▓{CY}█{CD}░{RS}     {MG}╱{RS}     {MB}◉{WT}│{MB}◉{RS}      {MG}╲{RS}      {CD}░{CY}█{CD}▓{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("MG", MG),
               fmt::arg("MB", MB), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "   {CD}▒{CY}█{CD}░{RS}    {WT}M{MG}━━━━━━━━━━━━━━━━{WT}T{RS}     {CD}░{CY}█{CD}▒{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD),
               fmt::arg("MG", MG), fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "     {CD}░▓{CY}██░{RS}         {WT}P{RS} {MB}·{RS} {WT}T{RS}        {CD}░{CY}██{CD}▓░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("MB", MB),
               fmt::arg("WT", WT), fmt::arg("RS", RS));
    fmt::print(stderr, "     {CD}░▒▓{CY}██████████████{CD}▓▒░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("RS", RS));
    fmt::print(stderr, "        {CD}░▒▓{CY}██████████{CD}▓▒░{RS}\n",
               fmt::arg("CY", CY), fmt::arg("CD", CD), fmt::arg("RS", RS));
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

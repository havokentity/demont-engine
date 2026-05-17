# Testing strategy

Living doc. Codifies what we test, why we test it, and the order we
build it in. Updated as phases land.

## Goal

Catch the bug classes we've actually hit in development before they
ship, with minimum dev-hour investment. Optimize for "wrong pixels
in a specific backend / denoiser / scene combo" because that's the
shape of nearly every regression we've seen.

## Bug classes we're optimizing for

Documented because every test investment should be traceable to a
specific class of failure we've hit (or know we'd hit if we shipped
without a test).

| Bug | Detection today | Detection with this plan |
|---|---|---|
| Cbuffer alignment regression (Engine.cpp PtPush, May 2026) | Hours of eyeballing on a specific (Metal, svgf_basic) combo | Golden-image diff on first push |
| CSG silently dropped on Mac-Vulkan (issue #44) | Surfaced by accident during MoltenVK bringup | Golden-image diff on (Vulkan, off, cornell_csg) |
| MoltenVK nullDescriptor device-create gate (PR #43) | First Mac-Vulkan launch ever | Smoke test asserts first frame renders |
| `/U DEBUG` CMake list-split (earlier session) | Windows CI build failure (matrix coverage) | Already covered by Windows matrix |
| Cross-host `FETCHCONTENT_BASE_DIR` collision (PR #41) | Copilot review | Already covered by Copilot |

Pattern: most regressions are pixel-correctness, build-system, or
platform-specific init issues. Almost none are pure-logic bugs in
C++ that a traditional unit test would catch. The investment ladder
reflects that.

## Tier 1 -- highest ROI, low infra cost

### 1.1 Golden-image regression matrix (issue #45)

For each `(backend, denoiser_kind, scene)` combo, render one frame
headlessly and assert pixel match within tolerance against a stored
reference image.

**MVP shipped** (initial PR for #45): 1 scene x 3 backends x 1
denoiser = 3 cells. Harness, comparison tool, CI integration, and
the first Mac goldens are in. The matrix is generic over (scene,
backend, denoiser) so further cells expand via additional
`pt_add_golden_cell(...)` rows -- see "How to add a new test"
below. Initial Mac goldens (Darwin host) are committed; the
Windows golden_software baseline needs to be regenerated on a
Windows box and committed in a follow-up. There is no Linux
golden CI today -- `pt_add_golden_cell` skips the `software` cell
on non-(APPLE OR WIN32) hosts (rhi_software isn't ported to Linux
yet), and the Linux sanitizers workflow does not run the matrix.

**Target matrix** (per the issue): **18 backend x denoiser cells**
(1 Software + 6 Metal + 3 Vulkan + 8 Vulkan-OptiX) x **3 scenes**
(`cornell_csg`, `hdri_env`, `procedural_night`). Tracked in #45
follow-ups.

**Architecture** (as implemented):
- `tools/pt_render_one_frame/` -- thin C++ harness CLI that
  translates `(scene, backend, denoiser, spp, frames, out)` into
  the engine's smoke-test plumbing and fork/exec's `demont`. No
  GPU / engine library links so it builds in ms.
- `src/engine/Engine.cpp` -- two new cvars + CLI flags wired into
  the existing `--smoke-frames` loop:
    - `pt_smoke_exec` (`--smoke-exec=path`): exec a console-script
      fixture right after demont.cfg + autoexec.cfg + CLI overrides
      load but before the device boots.
    - `pt_smoke_capture_out` (`--smoke-capture-out=path`):
      synchronous final-frame readback + host-side ACES + sRGB
      OETF + stbi_write_png after the smoke loop hits its budget,
      before TearDownDevice releases the GPU textures.
- `tools/imgdiff/` (existing, PR #92) -- per-pixel L2 thresholded
  diff with --max-delta / --mean-delta / --fail-percent gates and
  a colorised diff PNG.
- `tests/goldens/scenes/<scene>.cfg` -- deterministic fixture
  pinning resolution, camera, sun, spp, accumulator EMA, capture
  seed.
- `tests/goldens/<host>/<scene>__<backend>__<denoiser>.png` --
  per-host committed reference PNGs.
- `tests/CMakeLists.txt` `pt_add_golden_cell(...)` helper -- one
  ctest pair per cell (`_render` + `_diff`), skipped at configure
  time when the backend isn't available on the host.
- `.github/workflows/build.yml` -- per-host golden step that
  runs the `golden_software` label subset (only backend that
  works on GH hosted runners) and uploads `golden_actual/` +
  `golden_diff/` as workflow artefacts on failure.

Would have caught both bug class entries 1 + 2 above the moment
they landed.

**Estimate:** 5 dev-days for the harness + diff tool + CI wiring,
plus 1 day per host for initial goldens.

### 1.2 Push-constant layout invariants

Compile-time `static_assert` patterns on every push struct:

1. `static_assert(sizeof(X) % 16 == 0)` -- the struct itself is
   16-byte aligned end-to-end.
2. `static_assert(offsetof(X, vec4_field) % 16 == 0)` -- every
   `float4` / `uint4` field starts on a 16-byte boundary, mirroring
   the std140 / MSL cbuffer rule the Slang compiler applies on the
   GPU side.
3. `static_assert(sizeof(X) == <sum>)` -- if the struct mirrors a
   shader cbuffer, a byte-exact size check catches drift.

PtPush already has all three (added in `18498f0`). TonePush has
the size + offsetof checks (`sizeof == 624`, `offsetof(ghosts) ==
112`). Phase 0 audits the rest of the push structs in the codebase
and adds defensive asserts where they're missing.

**Estimate:** half a day.

### 1.3 Backend smoke tests + validation-layer gating

A `demont --smoke-test --backend X --frames 32 --output result.png`
mode that:

1. Initializes the requested backend.
2. Loads a fixed test scene.
3. Renders 32 frames headlessly.
4. Captures the final frame to PNG.
5. Exits 0 on success, non-zero on any error / crash / validation
   warning.

Plus:

- **Vulkan validation layers ON in smoke-test mode** -- any
  `VK_DEBUG_REPORT_WARNING` or `VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT`
  exits non-zero. Catches descriptor-binding regressions,
  layout-mismatch regressions, and synchronization regressions
  before they ship. The existing `DebugCallback` in
  `VulkanDevice.cpp:114` already routes severity -- we just need to
  raise an exit flag instead of `return VK_FALSE`.
- **Metal API validation ON** -- same idea via
  `MTL_DEBUG_LAYER=1` env var.
- **Software backend** just runs.

Catches the class of bug that PR #43 fixed (device-create
regressions), the class issue #44 covers (TLAS / BLAS / RayQuery
regressions), and any future descriptor / binding regression.

**Estimate:** 2 dev-days.

## Tier 2 -- meaningful ROI, more infra

### 2.1 Unit tests for non-GPU hot paths

Modules where deterministic input -> deterministic output, no
device:

| Module | What we'd test |
|---|---|
| `pt_math` | vec/mat/quat ops, transform composition, basis vectors |
| `pt_csg/CsgScene` | Manifold ops correctness, transform stack |
| `pt_renderer/Astronomy` | Sun/moon position vs JPL Horizons reference; J2000 rotation; sidereal time |
| `pt_renderer/HdrImage` | CDF building from a known small input |
| `pt_console/Console` | cvar parse/set/serialize round-trip; cfg load/save |
| `pt_core/AnalyticBvh` | Builder + traversal vs naive linear scan on a known input |
| Push struct layouts | Already in 1.2 |

**Framework:** `doctest` (single-header, ~5x faster compile than
Catch2/GTest, MIT, supports `TEST_CASE`/`SUBCASE`/`CHECK`/`REQUIRE`).

**Estimate:** 3-4 dev-days for initial coverage.

### 2.2 AddressSanitizer + UBSan build on CI

New CMake preset `linux-asan-ubsan-debug` with
`-fsanitize=address,undefined -fno-sanitize-recover=all`. Runs:

- All unit tests under sanitizer instrumentation

Smoke-test + lo-spp goldens deliberately omitted from the Linux
sanitizer job: `ubuntu-latest` has no GPU and no usable Vulkan ICD
(see PR #77 for the full investigation -- engine-side smoke mode is
ready to re-wire when a self-hosted GPU runner appears). The Linux
software backend doesn't exist yet either (`src/CMakeLists.txt` gates
`rhi_software` to Apple/Windows; a Linux present path is a separate
work item). So the CI gate exercises the doctest test suite under
ASan/UBSan -- which is what the project's pre-existing tests already
target: CPU code paths like cvar parsing, capture-format selection,
image-diff compute, and (post Phase 3 unit-test landings) pt_math /
pt_csg / Astronomy / HdrImage / AnalyticBvh / Console.

Catches: OOB writes (would have caught a misaligned write if it
crossed an allocated boundary), use-after-free, leaks, signed
overflow, alignment violations. Linux runner because clang's ASan
is more reliable there than on macOS-arm64.

**How to reproduce locally on a Linux box**

```bash
# One-shot configure + build + test under sanitizers
cmake --preset linux-asan-ubsan-debug
cmake --build build/linux-asan-ubsan-debug --parallel
ctest --test-dir build/linux-asan-ubsan-debug --output-on-failure
```

The preset disables `PT_ENABLE_VULKAN_BACKEND` + `PT_ENABLE_OPTIX` so
no Vulkan SDK or CUDA install is required; Embree is auto-fetched as
a prebuilt tarball via `cmake/EmbreeBinary.cmake`. To inspect
sanitizer-handler env vars (mirroring CI):

```bash
export ASAN_OPTIONS="halt_on_error=1:detect_leaks=1:detect_stack_use_after_return=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
ctest --test-dir build/linux-asan-ubsan-debug --output-on-failure -V
```

UBSan sub-checks `vptr` and `function` are opted out of project-wide
in `cmake/CompilerWarnings.cmake`: both require RTTI, which this
project disables via `-fno-rtti` (orthodox-C++23 policy). All other
UBSan sub-checks (signed-integer overflow, alignment, null deref,
unreachable, vla-bound, shift, etc.) stay on.

**How to read a sanitizer report**

ASan reports look like:

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
READ of size 4 at 0x... thread T0
    #0 0x... in pt::csg::FlattenScene(...) src/csg/CsgScene.cpp:142:18
    #1 ...
```

The first `#0` frame after the `ERROR:` line is the offending site;
follow the stack up to find your code. `READ` vs `WRITE` tells you
the direction.

UBSan reports look like:

```
src/math/Quat.cpp:88:14: runtime error: signed integer overflow:
  2147483647 + 1 cannot be represented in type 'int'
```

File:line:col is the exact source location. UBSan tags
`-fno-sanitize-recover=all` so the run also exits non-zero on the
first hit (the CI job's `halt_on_error=1` env var enforces the same
on the ASan side).

**Suppressions**

If a third-party shutdown leak (Embree's TBB worker pool draining,
mimalloc's segment-cache final-release, etc.) trips LSan with a
clean-stack-frame report, add a narrow rule to `.lsan-suppressions.txt`
at the repo root. **Do not blanket-suppress** -- each rule should
name the specific stack frame so a real leak introduced later in the
same module still surfaces. The file is wired in via the CI job's
`LSAN_OPTIONS=suppressions=...:print_suppressions=0` env var.

**Estimate:** 1 dev-day.

### 2.3 Cvar persistence round-trip

Single test: load `demont.cfg`, dump current state to a string,
re-load, assert identical. Catches cfg-format regressions silently
today.

**Estimate:** 2 hours.

## Tier 3 -- nice to have, defer

- **Shader compile-only tests** -- already implicit in build, low
  marginal value.
- **Performance regression** -- noisy, fragile, explicitly out of
  scope per #45.
- **Stress tests** -- large scenes, long renders. Defer.
- **Mutation testing** -- pure ROI exercise.
- **Fuzzing** -- narrow surface area. Defer.

## Feature gates (orthogonal to testing but related)

These are platform-correctness features that the testing matrix
exists to verify. Filed as their own work items because they're
code changes, not test changes.

### Platform-aware cvar autocompletion

Today the console's `r_denoiser` tab-completion offers every value
the cvar accepts -- including `optix_*` on Mac (where OptiX requires
NVIDIA + CUDA + Windows) and `metalfx*` on Linux / Windows (where
MetalFX doesn't exist). This is a UX bug today and a
visual-correctness bug tomorrow when someone selects an unsupported
value and the engine falls back silently.

Fix shape: make the cvar's `allowed_values` list dynamic at runtime
based on platform + active backend + build-time `PT_ENABLE_OPTIX`.
Console completion reads from `allowed_values`, so this fix
propagates automatically.

Other cvars likely with the same issue: `r_backend` (Metal Mac-only,
Vulkan everywhere with-MoltenVK-or-driver, Software always),
`r_sky_mode` (might be fine -- all three modes are cross-platform),
anything that hard-codes a CUDA / OptiX / MetalFX value.

Filed as its own issue, must-have feature.

## Tooling choices

| Concern | Pick | Rationale |
|---|---|---|
| Unit test framework | doctest | Single-header, MIT, fast compile, drop-in like Catch2 |
| Image diff | Hand-rolled `imgdiff` (~200 LOC) | No good lightweight C++ image-diff lib |
| PNG I/O | `stb_image_write` / `stb_image` (already vendored) | Already in deps |
| Test runner | `ctest` (CMake built-in) | Standard, GH Actions integrates natively |
| Mocking / fakes | None needed | Software backend already serves as "fake GPU" |
| Vulkan validation | Loader's built-in | No extra deps |
| Metal validation | `MTL_DEBUG_LAYER=1` env | No extra deps |
| ASan/UBSan | Clang's built-in | No extra deps |

## CI matrix evolution

```
Today:
  mac-26                (build + zero-warnings + unit tests)
  windows-latest        (build + unit tests)
  linux-asan-ubsan      (build + unit tests under ASan/UBSan)  <-- Phase 3 [#68, this PR]

Target after Phase 2 (golden matrix #45):
  mac-26                (build + unit + smoke + goldens)
  windows-latest        (build + unit + smoke + goldens)
  linux-asan-ubsan      (build + unit under sanitizers; smoke + goldens wait
                         for a self-hosted GPU runner or GH GPU-tier OSS access)
  linux-release         (build + software-only goldens, once a Linux software
                         backend present path lands)
```

Free for public repos on GH Actions' standard `ubuntu-*` /
`macos-*` / `windows-*` runners -- CI cost is not a concern.

## Phased execution

```
Phase 0 (now, ~half day):           Push-constant layout audit + asserts  [1.2]
                                    Pure compile-time, lands fast.

Phase 1 (week 1, ~5 days):          Test framework + smoke tests
                                    - doctest as a CMake dep
                                    - tests/ directory + ctest wiring
                                    - Backend smoke tests              [1.3]
                                    - Validation-layer gating          [1.3]
                                    - Cvar round-trip                  [2.3]

Phase 2 (week 2-3, ~7 days):        Golden-image matrix                [1.1 / #45]
                                    - pt_render_one_frame harness        [done]
                                    - imgdiff tool                       [done, PR #92]
                                    - CI integration (mac, win, lin)     [MVP: software cell per host]
                                    - Initial goldens                    [Darwin done; Win + Linux
                                                                          via per-host regen workflow]
                                    - Matrix expansion                   [follow-up PRs: SVGF cells,
                                                                          hdri_env scene, procedural_night,
                                                                          OptiX denoiser cells on RTX hw]

Phase 3 (week 3-4, ~5 days):        Unit tests + sanitizers            [2.1 + 2.2]
                                    - pt_math, pt_csg, Astronomy
                                    - HdrImage, AnalyticBvh
                                    - Linux ASan+UBSan job

Phase 4 (parallel, ~1 day):         Platform-aware cvar autocompletion
                                    - Dynamic allowed_values per cvar
                                    - r_denoiser, r_backend, anything else
                                      with platform-conditional valid sets

Phase 5 (later):                    Tier 3 as appetite allows
```

Total Phase 0-3: ~3 weeks of focused work.

## How to add a new test

The Phase 1 + 2 harness is wired and a contributor adds tests
without touching the engine.

### Doctest-based unit tests

1. Drop `tests/<area>/<name>_test.cpp` (or `tests/<name>_test.cpp` for
   shared helpers like `harness_smoke`). First two lines:
   ```cpp
   #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
   #include <doctest/doctest.h>
   ```
2. Add a `pt_add_test(<name> <source.cpp> [LINK_TARGETS ...])` line
   under the existing block in `tests/CMakeLists.txt`. The helper
   wires doctest, the project's strict warning policy, and ctest
   registration with a 60 s timeout.
3. Run `ctest --test-dir build/<preset>` to confirm.

### Golden-image cells

Each cell is one `pt_add_golden_cell(SCENE ... BACKEND ... DENOISER
...)` row in `tests/CMakeLists.txt` (under the "Golden-image
regression matrix" banner). Required args:

- `SCENE`     -- name of a fixture at `tests/goldens/scenes/<name>.cfg`.
- `BACKEND`   -- one of `software` / `metal` / `vulkan`. Cells whose
  backend isn't available on the build host (Metal off Apple, Vulkan
  when `PT_ENABLE_VULKAN_BACKEND=OFF`) are silently skipped at
  configure time.
- `DENOISER`  -- value the cell sets `r_denoiser` to via the harness'
  synthesised pre-exec script.

Optional knobs: `FRAMES` (smoke-test frame budget, default 64),
`SPP`, `EXTRA` (semicolon-separated console-script lines inlined
into the pre-exec), and the imgdiff thresholds `MAX_DELTA` /
`MEAN_DELTA` / `FAIL_PERCENT`. Per-pixel L2 above MAX_DELTA marks
a pixel as "bad"; a cell fails if mean L2 > MEAN_DELTA or
bad-pixel ratio > FAIL_PERCENT %.

### Adding a new scene fixture

1. Drop `tests/goldens/scenes/<scene>.cfg`. **Cvar writes only**
   in the current implementation -- the smoke-exec pass runs in
   `Engine::Init` BEFORE `csg_scene_` is constructed and BEFORE
   `SeedDefaultPrimitives()` runs (see `src/engine/Engine.cpp`
   around the `pt_smoke_exec` exec). That means
   the CSG console commands (`csg_box`, `csg_sphere`, `csg_cylinder`,
   `csg_op`, `csg_reset`, `csg_remove`, `csg_dump`) and analytic-
   primitive commands (`prim_*`) are out of scope -- queued for a
   future late-phase exec (TODO(#45-followup) in Engine.cpp).
   Avoid `;` and backtick
   characters inside `#` comments -- the console parser splits
   scripts on `;` even mid-comment, which leaks the post-semicolon
   text as a fresh command.
2. Lock down everything the matrix needs to be deterministic:
   resolution (`app_window_width` / `_height`), camera pose
   (`cam_pos`, `cam_yaw`, `cam_pitch`, `cam_fov`), sun
   (`r_sky_mode`, `r_sun_*`), path-tracer (`r_spp`, `r_max_bounces`,
   `r_firefly_clamp`, `r_accum_ema_alpha`), exposure
   (`r_auto_exposure 0`, `r_exposure <fixed>`), post effects
   (`r_bloom 0`, `r_lens_flare 0` unless the cell is specifically
   testing them), capture format / seed (`r_capture_format png`,
   `r_capture_seed <non-zero>`).
3. Do NOT pin `r_backend`, `r_denoiser`, `pt_smoke_*` cvars in
   the fixture -- those are set by the harness wrapper per cell.

## How to regenerate goldens

Goldens live under `tests/goldens/<host>/<scene>__<backend>__<denoiser>.png`,
where `<host>` is `Darwin` / `Linux` / `Windows` (`CMAKE_SYSTEM_NAME`).
Each host owns its own subdirectory; cross-platform variance from
FP nondeterminism, driver versions, and resolution defaults
(Vulkan on Retina Mac renders at 2x logical size, for example) is
not something tolerance thresholds can absorb cleanly, so
per-host goldens stay the canonical pattern.

Regeneration is one command per cell:

```bash
# from the worktree root, after a clean build:
./build/<preset>/tools/pt_render_one_frame/pt_render_one_frame \
    --scene   tests/goldens/scenes/cornell_csg.cfg \
    --backend software \
    --denoiser off \
    --frames  64 \
    --out     tests/goldens/$(uname -s)/cornell_csg__software__off.png
```

For all-cell regen on a host:

```bash
# Mac (Darwin):
for b in software metal vulkan; do
    rm -f demont.cfg
    ./build/mac-debug/tools/pt_render_one_frame/pt_render_one_frame \
        --scene   tests/goldens/scenes/cornell_csg.cfg \
        --backend "$b" \
        --denoiser off \
        --frames  64 \
        --out     "tests/goldens/Darwin/cornell_csg__${b}__off.png"
done
```

After regen:

1. `git diff --stat tests/goldens/` -- verify only the expected PNGs
   changed. Surprises (extra files, an unrelated host's directory)
   are likely a working-dir slip-up.
2. Eyeball each new PNG against the prior commit's version. A
   legitimate visible change should be obvious; a regression should
   look broken.
3. Commit with a `test(goldens): refresh ... after ...` style
   message that says WHY they were regenerated.

CI uploads two artefact directories on every run -- `golden_actual/`
(what this PR's build produced) and `golden_diff/` (the imgdiff
heatmaps). On a failing cell, downloading the artefact ZIP and
opening the diff PNG points at the exact pixels that drifted.

## Maintenance + care of goldens

- **Stochastic noise floor.** The default 64 smoke frames at
  `r_spp 1` on the cornell_csg scene settles below the matrix's
  default thresholds (max-delta 16, mean-delta 2, fail-percent 1).
  Lower frame counts trip imgdiff with the Metal HW-RT BLAS build
  entropy alone. If a new scene is noisier, bump `FRAMES` rather
  than weakening tolerances -- tighter tolerances are what catch
  the bug class this matrix exists for.
- **Per-cell tolerance.** Override `MAX_DELTA` / `MEAN_DELTA` /
  `FAIL_PERCENT` on the `pt_add_golden_cell(...)` line when a
  specific scene has known irreducible variance (HDRI sun pixel,
  thin specular highlights). Don't widen defaults.
- **"Legitimate visual change" vs "regression".** Rule of thumb:
  if the changed pixels are localised to a feature the touching
  PR mentions, it's a legitimate update -- regen, commit. If the
  diff is scattered or hits regions the PR shouldn't touch, it's
  a regression. Use the diff PNG heatmap as the deciding view.
- **Repo size.** PNGs commit at modest 512x384 (3-5 KB compressed
  per image) so even a 100-cell matrix fits in ~500 KB. Vulkan on
  Mac Retina renders at 1024x768 which lands around 80 KB per
  golden -- still cheap. If a scene needs 1080p for any reason,
  that's an LFS-territory decision; consult before bumping.
- **Cross-host parity.** Two hosts producing pixel-identical PNGs
  for the same `(scene, backend, denoiser)` is a happy accident,
  not a guarantee. ULP drift across architectures and driver
  versions means each host's golden lives independently. CI gates
  each host on its own subdirectory.

## Related

- Issue #45 -- the test matrix infra
- Issue #44 -- CSG-on-Mac-Vulkan (will get a regression test once #45 lands)
- Issue #46 -- SVGF stars post-composite fix (gated on test infra for verification)
- PR #43 -- introduced the alignment bug class that motivates Phase 0
- `Raytracer Plan/FOLLOW_UPS.md` -- the queued-design-sketch sibling doc

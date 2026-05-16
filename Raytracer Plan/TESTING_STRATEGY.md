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

Initial matrix: **18 backend x denoiser cells** (1 Software + 6 Metal
+ 3 Vulkan + 8 Vulkan-OptiX) x **3 scenes** (`cornell_csg`,
`hdri_env`, `procedural_night`).

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

- All unit tests
- Software-backend smoke test
- Initial render of each scene at low spp

Catches: OOB writes (would have caught a misaligned write if it
crossed an allocated boundary), use-after-free, leaks, signed
overflow, alignment violations. Linux runner because clang's ASan
is more reliable there than on macOS-arm64.

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
  mac-release    (build only)
  windows-release (build only)

Target after Phase 3:
  mac-release           (build + unit + smoke + goldens)
  windows-release       (build + unit + smoke + goldens)
  linux-asan-ubsan      (build + unit + software-only smoke + lo-spp goldens)
  linux-release         (build + software-only goldens)
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
                                    - pt_render_one_frame harness
                                    - imgdiff tool
                                    - CI integration (mac, win, lin)
                                    - Initial goldens

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

(Filled in after Phase 1 lands. The harness + ctest wiring needs
to exist before we can document "add a test here, register it
there.")

## How to regenerate goldens

(Filled in after Phase 2 lands. Critical for operator usability --
without a clear regen workflow every legitimate renderer change
becomes a multi-day golden-shuffle. Likely shape: `ctest -R
render_matrix --regenerate-goldens` re-renders, prints a diff
summary, lets you commit the new PNGs.)

## Maintenance + care of goldens

(Filled in after Phase 2 lands. Stochastic noise floor, seed
choice, tolerance per scene, what counts as "legitimate visual
change" vs "regression" -- the judgment calls every golden-image
workflow eventually grows.)

## Related

- Issue #45 -- the test matrix infra
- Issue #44 -- CSG-on-Mac-Vulkan (will get a regression test once #45 lands)
- Issue #46 -- SVGF stars post-composite fix (gated on test infra for verification)
- PR #43 -- introduced the alignment bug class that motivates Phase 0
- `Raytracer Plan/FOLLOW_UPS.md` -- the queued-design-sketch sibling doc

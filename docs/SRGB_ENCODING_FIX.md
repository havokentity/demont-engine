# Vulkan swapchain sRGB encoding fix — deploy & test

> **HISTORICAL (2026-07-08):** this fix shipped with the Windows
> bringup (PR #1, merged 2026-05-08) and lives on `main`; the
> `feature/windows-nvidia` branch referenced below has been deleted.
> Kept as deploy/test archeology — for a fresh checkout just build
> `main`.

> Commit landed. Pull, build, test, decide.

## TL;DR for the user

> demont was silently broken on Vulkan — the swapchain was missing the
> linear→sRGB encode step that Mac/Metal does for free in hardware. Win
> output ended up with the OS double-decoding linear values as if they
> were already sRGB-encoded, making everything dimmer than the engine
> intended. PPMs matched between backends because PPMs go through a
> separate, correct, host-side tonemap; only the on-screen render path
> was affected. Fix is a manual sRGB OETF inside the Slang shaders,
> gated to the SPIR-V target so Mac stays untouched.

## What landed

Commit on `feature/windows-nvidia` adds `srgb_oetf()` to PathTrace.slang
and Tonemap.slang, gated by `#if defined(PT_TARGET_SPIRV)`, applied at
every swapchain-write site:

- `shaders/PathTrace.slang`: helper at the top of the file (next to
  `acesTonemap`), and `output[tid] = float4(srgb_oetf(ldr), 1.0)`
  at the end of `main`. The `denoise_color` write inside the
  `if (denoiser_enabled)` branch is **deliberately** left linear —
  it writes to a separate RGBA16F intermediate that Tonemap reads
  later, must stay linear.
- `shaders/Tonemap.slang`: helper near the top, `PT_SRGB_ENCODE(x)`
  macro that expands to `srgb_oetf(x)` on SPIR-V and `(x)` on Metal,
  applied at the three `acesTonemap(c * exposure_state[0])` sites
  (physical flare branch, sun flare branch, main).

Mac path: zero behaviour change, the macro is a no-op on Metal.

Vulkan path: linear ACES output now gets sRGB-encoded before the
swapchain store, so the OS's display-side EOTF reproduces the original
linear correctly instead of double-decoding.

## How to deploy

```sh
# Historical instructions -- the branch is merged + deleted; today this
# is just: git pull on main, then
cmake --build build/win-debug --parallel
.\build\win-debug\src\app\demont.exe
```

## What to expect visually

**Before fix**: demont on Win was visibly dimmer than Mac (especially
midtones). Highlights mostly fine, dark tones squashed.

**After fix**: demont on Win should look noticeably **brighter**, now
matching what `screenshot win.ppm` looks like in Photoshop — which is
also what Mac demont looks like onscreen. Side-by-side Mac and Win
should now be brightness-matched (modulo monitor calibration).

## Verification protocol

1. **Boot demont, eyeball the day scene.** Should look closer to Mac
   than before.

2. **Take fresh PPMs, compare:**
   ```
   screenshot win_day_after_fix.ppm
   ```
   Should still byte-match Mac's PPM at same scene/cvars (PPMs go
   through host-side tonemap, not the swapchain — so this stays
   unchanged from before the fix; it's a regression check that I
   didn't accidentally break the PPM path).

3. **Take an OS-level screenshot of demont running.** Compare to a
   Mac OS-level screenshot at the same scene/cvars. Now they should
   look the same brightness.

## ⚠️ Critical: NVIDIA might be auto-encoding behind our back

Vulkan spec only auto-encodes sRGB on **color-attachment** writes;
storage-image writes are documented as bypassing the format conversion.
**However**, some drivers (NVIDIA included historically) DO auto-encode
on storage writes despite the spec, as a "helpful" non-portable
behaviour. If that's the case here, this fix would **double-encode**
on the RTX (driver encodes once, shader encodes again) and the image
would look **too bright / washed out**.

If after deploying you see demont looking **nuked / blown out / too
bright**, that's the symptom of double-encoding. The fix in that case
is to revert the shader change and instead change the swapchain format
to `VK_FORMAT_B8G8R8A8_SRGB` — letting the driver do the encode it's
already doing.

Test plan if the shader fix looks blown out:

1. Revert the shader change locally (don't push):
   ```sh
   git checkout shaders/PathTrace.slang shaders/Tonemap.slang
   ```
2. In `src/rhi_vulkan/VulkanDevice.cpp:921`, change:
   ```cpp
   swap_format_ = VK_FORMAT_B8G8R8A8_UNORM;
   ```
   to:
   ```cpp
   swap_format_ = VK_FORMAT_B8G8R8A8_SRGB;
   ```
   And update the surface-format probe loop right below to prefer
   `VK_FORMAT_B8G8R8A8_SRGB`.
3. Rebuild + run.
4. If the image now looks correct (matches Mac), NVIDIA was indeed
   auto-encoding storage writes and the format change is what we want.
   Commit that as the fix instead of the shader change.

## How to confirm Mac stayed unchanged

Optional but worth doing once on the Mac side: take fresh PPMs +
OS-level screenshots on Mac after pulling the fix, compare to
pre-fix references. Should be byte-identical. If anything moves on
Mac, the `#if defined(PT_TARGET_SPIRV)` gating leaked somewhere and
needs investigation.

## Why this was missed for so long

The Tonemap.slang header comment (line 14–15) explicitly documented
the assumption: *"the linear-to-sRGB encode is implicit on store."*
That's true on Apple silicon for `BGRA8Unorm_sRGB`. It's untrue on
Vulkan generally and not even guaranteed by Vulkan spec for the
analogous `B8G8R8A8_SRGB` format on storage writes. When P12 ported
the engine to Vulkan it kept the assumption, picked `B8G8R8A8_UNORM`
without the `_SRGB`, and shipped. The PPMs hid the bug because the
PPM path bypasses the swapchain entirely.

User was right to push back on the `r_output_scale` workaround — that
would have masked this bug indefinitely.

## Don't do (post-mortem reminders)

- **Don't add `r_output_scale`.** The cvar would have applied a uniform
  dim to compensate for the missing OETF, but the OETF is non-linear,
  so the visual character would still be wrong (midtones especially).
  Real fix is the OETF, not a multiplier.
- **Don't change Mac's path.** It's correct (Apple silicon's
  storage-write encoding for `BGRA8Unorm_sRGB` works as expected).
  The macro / ifdef leaves it alone.
- **Don't switch Vulkan swapchain format alone** (without the shader
  fix) unless you've confirmed NVIDIA auto-encodes storage writes —
  per spec it shouldn't, so on a strict driver you'd still be writing
  un-encoded linear values into a SRGB-tagged image and the OS would
  still mis-interpret them. Test before relying.

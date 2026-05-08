# Vulkan swapchain sRGB encoding fix — handoff for Win Claude

> **Symptom**: demont's realtime output on Windows looks dimmer than on
> Mac. PPM screenshots from both backends are byte-identical (path
> tracer output is correct), but the on-screen image differs across
> backends regardless of monitor calibration / Win HDR mode. User
> intuition was correct: something is wrong on the Vulkan side.

## Diagnosis

The two backends configure the swapchain asymmetrically:

```
Mac  (Metal)  : MTLPixelFormatBGRA8Unorm_sRGB
                (Apple silicon auto-encodes linear -> sRGB on STORAGE
                 writes for sRGB pixel formats)

Win  (Vulkan) : VK_FORMAT_B8G8R8A8_UNORM + VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
                (no auto-encoding ever; OS treats stored bytes as already-
                 sRGB-encoded and applies EOTF to produce panel light)
```

Both PathTrace.slang's inline tonemap and Tonemap.slang write **linear**
ACES-tonemapped values directly to the swapchain, expecting the format
to handle encoding. From `shaders/Tonemap.slang:14`:

```
//   slot 1 (texture) : ldr_out -- swapchain (BGRA8Unorm_sRGB) -- the
//                       linear-to-sRGB encode is implicit on store.
```

That assumption holds on Mac. It doesn't on Vulkan.

What happens on Vulkan today:

```
shader writes linear 0.5
  -> stored raw to UNORM image (0.5 in memory)
  -> OS reads 0.5, applies sRGB EOTF (treats it as already encoded)
  -> EOTF(0.5) ~= 0.215 linear
  -> panel emits 0.215 linear instead of 0.5
  -> image looks DIMMER (most strongly in midtones; black + white roughly
     unchanged, but the curve in between is squashed).
```

Same on Mac:

```
shader writes linear 0.5
  -> Apple silicon's BGRA8Unorm_sRGB storage write encodes -> 0.735
  -> OS reads 0.735, applies sRGB EOTF -> 0.5 linear
  -> panel emits 0.5 linear (correct)
```

This is independent of Win HDR mode. With Win HDR off the Alienware
shows the dimmed image at panel SDR brightness; with Win HDR on the
DWM SDR-to-HDR remap further compresses to paper-white. Both Win paths
are dimmer than Mac because the underlying sRGB encoding step is
missing from the Vulkan write side.

## Vulkan spec evidence

From the Vulkan 1.3 spec on storage-image writes:

> Storage image writes do not perform format conversion for sRGB
> formats. Linear values written to a storage image with an sRGB
> format are stored as if the image had a non-sRGB UNORM format.

So even switching the swapchain format to `VK_FORMAT_B8G8R8A8_SRGB`
**will not fix this** — storage writes still bypass the encode. Only
color-attachment writes do the implicit encoding for SRGB formats.

## Fix options

### Option A: manual OETF in the shader, gated to SPIR-V — RECOMMENDED

Smallest change, fully portable across Vulkan drivers, doesn't touch
the Metal path. Apply the sRGB OETF (linear -> sRGB encode) in the
shader before storing to the swapchain output.

Slang snippet to drop into both `shaders/PathTrace.slang` and
`shaders/Tonemap.slang`:

```slang
// sRGB OETF (linear -> nonlinear). Standard piecewise definition
// from IEC 61966-2-1: linear segment near zero, gamma ~2.4 above.
float3 srgb_oetf(float3 c) {
    c = max(c, float3(0.0, 0.0, 0.0));
    float3 lin  = c * 12.92;
    float3 expo = 1.055 * pow(c, float3(1.0/2.4)) - 0.055;
    return select(c < 0.0031308, lin, expo);
}
```

Then guard the swapchain write:

```slang
float3 ldr = acesTonemap(avg * exposure);
#if defined(PT_TARGET_SPIRV)
    // Vulkan storage-image writes do not auto-encode sRGB even with
    // VK_FORMAT_B8G8R8A8_SRGB format -- spec only auto-encodes for
    // color-attachment writes. Apply OETF manually so the OS's
    // EOTF-on-display reproduces the original linear values.
    ldr = srgb_oetf(ldr);
#endif
output[tid] = float4(ldr, 1.0);
```

Three sites in Tonemap.slang (`acesTonemap(c * exposure_state[0])` x3),
one main site in PathTrace.slang's inline tonemap (`output[tid] = ...`).
The `denoise_color` write inside PathTrace.slang's denoiser-on path
should NOT get the OETF — it writes to a separate RGBA16F texture
that's later read by Tonemap.slang in linear space.

`PT_TARGET_SPIRV` is already defined by `cmake/Slang.cmake` for the
SPIR-V compile target; `PT_TARGET_METAL` for the MSL one. Existing
example: `shaders/PathTrace.slang:184` uses `#if defined(PT_TARGET_SPIRV)`
for the cbuffer Frame split.

**Pros**:
- Portable across every Vulkan driver (NVIDIA, AMD, Intel, Mesa)
- Mac path completely untouched (Metal's storage-write auto-encode
  on Apple silicon keeps doing its thing)
- Tiny diff (~10 lines across two shaders)
- Easy to A/B (toggle the `#if` to compare)

**Cons**:
- Extra ~5 ALU ops per output pixel (negligible at modern GPU rates)
- Mac stays dependent on Apple silicon's storage-write encoding
  behaviour, which isn't strictly Metal-spec-guaranteed but works
  reliably on M1/M2/M3/M4

### Option B: switch Vulkan to SRGB swapchain format + use color-attachment writes

The architecturally "correct" answer. Requires:

1. Change `VulkanDevice::RecreateSwapchain()` to prefer
   `VK_FORMAT_B8G8R8A8_SRGB` over `_UNORM`.
2. Change swapchain image USAGE to
   `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` (already has this) and stop
   using `VK_IMAGE_USAGE_STORAGE_BIT` for the swapchain output.
3. Add a graphics pipeline (vertex + fragment shader) that does a
   fullscreen-triangle pass reading the path tracer's output (now to
   an intermediate non-swapchain texture) and writes to the swapchain
   as a color attachment.
4. Modify PathTrace.slang to write to a separate output texture,
   not the swapchain directly.
5. Add a render pass + framebuffer for the final fullscreen pass.

**Pros**: clean architecture, matches the standard Vulkan SDR sRGB
pattern, no manual OETF in shaders.

**Cons**: 200+ LoC change, requires new vertex shader, new graphics
pipeline state, render-pass plumbing. Probably a separate PR. Not
worth the complexity for a single-line shader fix when Option A
solves the same problem.

### Option C: do nothing (ship as-is, document the asymmetry)

Could decide the dimmer Vulkan look is "fine, ship it." Argument
against: the user explicitly observed the difference, and the engine's
intent (linear ACES output through correct sRGB encoding to display)
is genuinely broken on Vulkan. Shipping a buggy color pipeline is bad
default. Don't pick this.

## Recommended action

**Land Option A.** Test plan:

1. Add `srgb_oetf()` helper to both `PathTrace.slang` and
   `Tonemap.slang`, gated on `PT_TARGET_SPIRV`.
2. Apply at all the swapchain-write sites:
   - `PathTrace.slang`: `output[tid] = float4(srgb_oetf_or_not(...), 1.0)`
   - `Tonemap.slang`: three `c * exposure_state[0]` sites at lines
     312, 372, 508
3. Skip the OETF on the `denoise_color` write inside PathTrace.slang
   (separate RGBA16F intermediate, must stay linear).
4. Build + test on the Win box. Demont should look noticeably brighter
   than before, and now match the Mac PPM-in-Photoshop reference.
5. If matching is good, take fresh `screenshot win_day.ppm` and
   compare to `mac_day.ppm` (should still be byte-identical since
   PPMs go through the engine's own tonemap, not the swapchain).
6. Take an OS-level screenshot of demont running and compare to
   Mac's OS-level screenshot. They should now look the same brightness
   regardless of monitor calibration (within calibration tolerance).

## Things to verify on the RTX

- Does NVIDIA's Vulkan driver actually auto-encode on storage writes
  to `VK_FORMAT_B8G8R8A8_SRGB`? Spec says no, but some drivers do.
  Test by switching to SRGB format with NO shader change; if image
  becomes correct, NVIDIA is being non-spec-helpful and Option A's
  shader OETF would double-encode (image too bright). If image stays
  the same dim look, NVIDIA follows spec and Option A is the right
  fix. **Verify before committing**.
- After Option A lands, confirm there's no double-encoding on Mac
  (the `#if defined(PT_TARGET_SPIRV)` should keep Metal untouched,
  but visual A/B still worth a glance).

## Why Mac was correct all along

Apple silicon's storage-write path for `BGRA8Unorm_sRGB` does the
linear -> sRGB OETF on store. This is documented Apple silicon
behavior (not strictly Metal-spec, but reliable on M1+). So Metal +
shader writing linear values + sRGB swapchain format -> correct.
Vulkan + shader writing linear + UNORM swapchain (or SRGB swapchain
with storage writes per spec) -> missing the OETF -> dim.

Engine code assumed the Metal behaviour was universal. It isn't.
Option A makes the assumption explicit and portable.

## Don't do

- Don't add an `r_output_scale` cvar to mask this. That dims Mac to
  hide the fact that Vulkan is wrong, instead of fixing Vulkan. The
  user already pushed back on this — correctly.
- Don't switch swapchain to SRGB format without also switching to
  color-attachment writes (Option B's full version). The format
  change alone won't fix anything for storage writes.
- Don't change the Metal path. It's already correct.

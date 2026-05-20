#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajesh D'Monte
#
# Wave 8 PBR (#26): generate the small procedural test-texture set used by
# tests/goldens/scenes/pbr_textured.cfg. Re-run to regenerate the PNGs:
#
#     python3 scripts/gen_test_textures.py
#
# Outputs (256x256 RGBA8 PNG) under assets/textures/:
#   checker_albedo.png    -- 8x8 magenta/teal checkerboard (sRGB-encoded
#                            colour; the path tracer decodes to linear).
#   normal_bumps.png      -- tangent-space normal map, a grid of rounded
#                            bumps. Encoded [-1,1] -> [0,1]; flat regions
#                            are (128,128,255) = +Z.
#   roughness_gradient.png-- horizontal 0..1 roughness ramp (linear; .r
#                            scales the material's flat roughness).
#
# Deterministic + dependency-light (numpy + Pillow, both already present
# in the dev env). The images are tiny so committing them is cheap and
# keeps the golden fixture self-contained without a build-time generator.

import os
import numpy as np
from PIL import Image

SIZE = 256
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "textures")


def save(name, rgb_or_rgba):
    arr = np.clip(rgb_or_rgba, 0, 255).astype(np.uint8)
    if arr.shape[2] == 3:
        alpha = np.full((SIZE, SIZE, 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    Image.fromarray(arr, "RGBA").save(os.path.join(OUT_DIR, name))
    print(f"wrote assets/textures/{name}  ({SIZE}x{SIZE} RGBA8)")


def checker_albedo():
    # 8x8 checkerboard. Two saturated sRGB colours so the lat/long + planar
    # UV mapping is obvious in the render.
    cells = 8
    cell = SIZE // cells
    a = np.array([220, 40, 200], dtype=np.float32)   # magenta
    b = np.array([40, 200, 190], dtype=np.float32)   # teal
    out = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    parity = ((xx // cell) + (yy // cell)) % 2
    out[parity == 0] = a
    out[parity == 1] = b
    return out


def normal_bumps():
    # Tangent-space normal map: a grid of rounded bumps. Build a height
    # field of cosine bumps, take its gradient, and encode the surface
    # normal (-dh/dx, -dh/dy, 1) normalized into [0,1].
    bumps = 6
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    u = xx / SIZE * bumps * 2.0 * np.pi
    v = yy / SIZE * bumps * 2.0 * np.pi
    h = 0.5 * (np.cos(u) * np.cos(v))            # height field in [-0.5,0.5]
    # Gradients (central-difference scale folds into the strength constant).
    strength = 2.0
    dhdx = -np.sin(u) * np.cos(v) * strength
    dhdy = -np.cos(u) * np.sin(v) * strength
    nx = -dhdx
    ny = -dhdy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    out = np.stack([(nx * 0.5 + 0.5) * 255.0,
                    (ny * 0.5 + 0.5) * 255.0,
                    (nz * 0.5 + 0.5) * 255.0], axis=2)
    return out


def roughness_gradient():
    # Horizontal 0..1 ramp, replicated down every row. Linear data map.
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    r = (xx / (SIZE - 1)) * 255.0
    out = np.stack([r, r, r], axis=2)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    save("checker_albedo.png", checker_albedo())
    save("normal_bumps.png", normal_bumps())
    save("roughness_gradient.png", roughness_gradient())


if __name__ == "__main__":
    main()

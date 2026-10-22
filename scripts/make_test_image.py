#!/usr/bin/env python3
"""Builds a synthetic dim, grainy test photo under samples/.

Not trying to be photorealistic - just dark, a little noisy, and with
enough edges/shapes that the CV steps have something real to react to.
Useful for a quick sanity check without needing an actual night-time
photo checked into the repo.

    python scripts/make_test_image.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
WIDTH, HEIGHT = 640, 480


def build_base_scene(harsh: bool) -> np.ndarray:
    y_idx, x_idx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)

    # a single warm light source, upper-left, falling off toward the edges
    light_x, light_y = WIDTH * 0.3, HEIGHT * 0.25
    dist = np.sqrt((x_idx - light_x) ** 2 + (y_idx - light_y) ** 2)
    falloff = np.clip(1.0 - dist / (WIDTH * 0.9), 0.0, 1.0) ** 1.5

    if harsh:
        # barely-lit alley at night: dim source, almost nothing in the shadows
        base_lift = (20, 18, 15)
        glow = (55, 60, 70)
    else:
        base_lift = (60, 55, 45)
        glow = (90, 100, 130)

    scene = np.stack(
        [
            base_lift[0] + falloff * glow[0],  # B
            base_lift[1] + falloff * glow[1],  # G
            base_lift[2] + falloff * glow[2],  # R - warm lamp glow
        ],
        axis=-1,
    )

    # a handful of dark rectangular silhouettes so there's edge structure,
    # like furniture sitting in the shadows
    rng = np.random.default_rng(42)
    for _ in range(6):
        x0 = int(rng.integers(0, WIDTH - 80))
        y0 = int(rng.integers(0, HEIGHT - 60))
        w = int(rng.integers(40, 140))
        h = int(rng.integers(30, 100))
        shade = rng.uniform(0.4, 0.75)
        scene[y0 : y0 + h, x0 : x0 + w] *= shade

    return scene


def add_sensor_grain(image: np.ndarray, rng: np.random.Generator, harsh: bool) -> np.ndarray:
    grain_std = 26 if harsh else 14
    blotch_std = 11 if harsh else 6

    noise = rng.normal(0, grain_std, image.shape).astype(np.float32)
    noisy = image + noise

    # low-frequency color blotching, closer to how real high-ISO shadow
    # noise looks than pure per-pixel grain
    blotches = rng.normal(0, blotch_std, image.shape[:2]).astype(np.float32)
    noisy[..., 0] += blotches
    noisy[..., 1] += blotches * 0.6

    return noisy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harsh",
        action="store_true",
        help="generate a much darker, noisier sample (samples/dark_alley.jpg) instead of the default one",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(7 if not args.harsh else 13)
    scene = build_base_scene(harsh=args.harsh)
    noisy = add_sensor_grain(scene, rng, harsh=args.harsh)
    final = np.clip(noisy, 0, 255).astype(np.uint8)

    output_path = SAMPLES_DIR / ("dark_alley.jpg" if args.harsh else "dim_room.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)
    print(f"wrote {output_path} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()

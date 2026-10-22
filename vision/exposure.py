"""
Brightness / exposure related image operations.

Everything here works on BGR uint8 arrays (the native OpenCV layout) so we're
not constantly converting color spaces back and forth between modules.
"""

import cv2
import numpy as np


def brightness_stats(image: np.ndarray) -> dict:
    """Cheap histogram-based read on how dark/bright/flat an image is.

    Returns plain floats/ints so this can be dropped straight into a report
    or a prompt without extra formatting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())

    total_px = gray.size
    dark_px = int(np.sum(gray < 60))
    bright_px = int(np.sum(gray > 235))

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "percent_dark": round(100 * dark_px / total_px, 2),
        "percent_clipped_bright": round(100 * bright_px / total_px, 2),
    }


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Standard gamma correction via a lookup table.

    gamma < 1 brightens shadows, gamma > 1 darkens. We only ever push
    gamma below 1 in this project since we're lifting *low*-light images,
    but the function itself is neutral.
    """
    gamma = max(0.1, gamma)
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    Done in LAB space so we only touch lightness and leave color/saturation
    alone - running CLAHE on each BGR channel separately tends to shift
    color balance in a way that looks off.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

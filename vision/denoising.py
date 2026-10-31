
"""
Denoising operations, ordered roughly best-quality-first.

Low-light shots are prone to high ISO grain and color noise.
The functions below provide multiple denoising methods with
resolution-aware processing for large images.
"""

import cv2
import numpy as np


def estimate_noise(image: np.ndarray) -> float:
    """Estimate image noise using Laplacian variance."""
    if image is None:
        raise ValueError("Input image cannot be None")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return float(laplacian.var())


def _get_search_window(image: np.ndarray) -> int:
    """Select NL-means search window based on image resolution."""
    height, width = image.shape[:2]
    pixels = height * width

    if pixels <= 1_000_000:
        return 21
    elif pixels <= 4_000_000:
        return 15
    else:
        return 11


def denoise_nlmeans(
    image: np.ndarray,
    strength: float = 10.0
) -> np.ndarray:
    """
    Non-local means denoising.

    Uses smaller search windows for larger images to reduce
    processing time without resizing or cropping the image.
    """
    if image is None:
        raise ValueError("Input image cannot be None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")

    original_shape = image.shape

    strength = float(np.clip(strength, 3, 30))
    search_window = _get_search_window(image)

    result = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        strength,
        strength,
        7,
        search_window
    )

    if result.shape != original_shape:
        raise RuntimeError("Denoising changed image dimensions")

    return result


def denoise_bilateral(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """Edge-preserving smoothing; first fallback method."""
    if image is None:
        raise ValueError("Input image cannot be None")

    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )


def denoise_gaussian(
    image: np.ndarray,
    ksize: int = 5
) -> np.ndarray:
    """Gaussian smoothing; last-resort fallback."""
    if image is None:
        raise ValueError("Input image cannot be None")

    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(
        image,
        (ksize, ksize),
        0
    )

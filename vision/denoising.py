"""
Denoising operations, ordered roughly best-quality-first.

Low-light shots are the worst case for noise (high ISO grain, blotchy
color noise in the shadows) so this is really the core of the project.
The agent layer decides *which* of these to call and picks the fallback
if the preferred one throws or clearly makes things worse.
"""

import cv2
import numpy as np


def estimate_noise(image: np.ndarray) -> float:
    """Rough noise estimate using the variance of the Laplacian.

    Not a rigorous noise metric, but it's cheap, has no external
    dependencies, and correlates well enough with "grainy vs clean" to
    drive decisions here.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def denoise_nlmeans(image: np.ndarray, strength: float = 10.0) -> np.ndarray:
    """Non-local means. Best quality, keeps edges reasonably intact, but
    the slowest of the three and occasionally chokes on very large images
    or unusual pixel formats - hence the fallback chain in the agent."""
    strength = float(np.clip(strength, 3, 30))
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def denoise_bilateral(image: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving smoothing. Faster and more forgiving than NL-means,
    used as the first fallback."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def denoise_gaussian(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Plain Gaussian blur. Softens noise but blurs detail too - this is
    the last-resort fallback, only used if the two above both fail."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

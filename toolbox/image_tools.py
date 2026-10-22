"""Image loading/saving, plus the default registry every agent shares."""

from pathlib import Path

import cv2
import numpy as np

from toolbox.registry import ToolRegistry
from vision import denoising, exposure, sharpening
from vision.quality_metrics import score_image


def load_image(path: str) -> np.ndarray:
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image at {path} - check the path and that it's a valid image file")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), image)
    if not ok:
        raise IOError(f"cv2 refused to write to {out_path}")


def build_default_registry(use_mcp: bool = False) -> ToolRegistry:
    """Every enhancement primitive the agents are allowed to use, registered
    under a stable name. If you add a new vision op, register it here and
    it's immediately available to the planning/denoise agents."""
    registry = ToolRegistry()

    registry.register(
        "brightness_stats",
        exposure.brightness_stats,
        "Read mean brightness, spread, and clipping percentages from an image.",
    )
    registry.register(
        "estimate_noise",
        denoising.estimate_noise,
        "Rough Laplacian-variance noise estimate for an image.",
    )
    registry.register(
        "apply_gamma",
        exposure.apply_gamma,
        "Gamma-correct an image; gamma < 1 lifts shadows.",
    )
    registry.register(
        "apply_clahe",
        exposure.apply_clahe,
        "Adaptive local contrast boost done in LAB space (lightness only).",
    )
    registry.register(
        "denoise_nlmeans",
        denoising.denoise_nlmeans,
        "Non-local means denoising. Best quality, slowest, preferred choice.",
    )
    registry.register(
        "denoise_bilateral",
        denoising.denoise_bilateral,
        "Edge-preserving denoise. Faster fallback if NL-means fails.",
    )
    registry.register(
        "denoise_gaussian",
        denoising.denoise_gaussian,
        "Plain Gaussian blur. Last-resort denoise fallback.",
    )
    registry.register(
        "unsharp_mask",
        sharpening.unsharp_mask,
        "Light sharpening pass to recover detail lost during denoising.",
    )
    registry.register(
        "evaluate_quality",
        score_image,
        "Compare original and enhanced images using deterministic quality metrics.",
    )

    if use_mcp:
        from toolbox.mcp_registry import MCPToolRegistry
        return MCPToolRegistry(registry)
    return registry

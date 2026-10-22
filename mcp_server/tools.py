"""Deterministic image-processing tools exposed by the MCP server."""
import base64
from io import BytesIO

import cv2
import numpy as np

from vision import denoising, exposure, sharpening
from vision.quality_metrics import score_image


def decode_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64, validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError("image_base64 must contain a valid base64-encoded image") from exc
    if image is None:
        raise ValueError("decoded image is not a supported image")
    return image


def encode_image(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode processed image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def brightness_stats(image_base64: str) -> dict:
    return exposure.brightness_stats(decode_image(image_base64))


def estimate_noise(image_base64: str) -> float:
    return float(denoising.estimate_noise(decode_image(image_base64)))


def apply_gamma(image_base64: str, gamma: float) -> dict:
    if not 0.1 <= float(gamma) <= 3.0:
        raise ValueError("gamma must be between 0.1 and 3.0")
    return {"image_base64": encode_image(exposure.apply_gamma(decode_image(image_base64), float(gamma)))}


def apply_clahe(image_base64: str, clip_limit: float = 2.0, tile_grid: int = 8) -> dict:
    if not 0.1 <= float(clip_limit) <= 10.0 or not 2 <= int(tile_grid) <= 32:
        raise ValueError("clip_limit must be 0.1..10 and tile_grid 2..32")
    image = exposure.apply_clahe(decode_image(image_base64), clip_limit=float(clip_limit), tile_grid=int(tile_grid))
    return {"image_base64": encode_image(image)}


def denoise_nlmeans(image_base64: str, strength: float = 7.0) -> dict:
    if not 1 <= float(strength) <= 30:
        raise ValueError("strength must be between 1 and 30")
    return {"image_base64": encode_image(denoising.denoise_nlmeans(decode_image(image_base64), strength=float(strength)))}


def denoise_bilateral(image_base64: str, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> dict:
    if not 1 <= int(diameter) <= 15:
        raise ValueError("diameter must be between 1 and 15")
    image = denoising.denoise_bilateral(decode_image(image_base64), diameter=int(diameter), sigma_color=float(sigma_color), sigma_space=float(sigma_space))
    return {"image_base64": encode_image(image)}


def denoise_gaussian(image_base64: str, ksize: int = 5) -> dict:
    if int(ksize) < 1 or int(ksize) > 31 or int(ksize) % 2 == 0:
        raise ValueError("ksize must be an odd integer between 1 and 31")
    return {"image_base64": encode_image(denoising.denoise_gaussian(decode_image(image_base64), ksize=int(ksize)))}


def unsharp_mask(image_base64: str, amount: float = 0.4) -> dict:
    if not 0 <= float(amount) <= 2:
        raise ValueError("amount must be between 0 and 2")
    return {"image_base64": encode_image(sharpening.unsharp_mask(decode_image(image_base64), amount=float(amount)))}


def evaluate_quality(original_base64: str, enhanced_base64: str) -> dict:
    """Evaluate two images using the project's deterministic quality metrics."""
    result = score_image(decode_image(original_base64), decode_image(enhanced_base64))
    return {
        "scores": result.get("scores", {}),
        "overall": float(result.get("overall", 0.0)),
        "before_stats": result.get("before_stats", {}),
        "after_stats": result.get("after_stats", {}),
        "before_noise": float(result.get("before_noise", 0.0)),
        "after_noise": float(result.get("after_noise", 0.0)),
    }

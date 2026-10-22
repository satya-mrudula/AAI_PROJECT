"""Detail recovery. Denoising almost always softens the image a bit, so we
follow it with a light unsharp mask to bring perceived detail back without
re-introducing the noise we just removed."""

import cv2
import numpy as np


def unsharp_mask(image: np.ndarray, amount: float = 0.6, radius: int = 5) -> np.ndarray:
    radius = radius if radius % 2 == 1 else radius + 1
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

"""
No-reference quality scoring.

There's no ground-truth "correct" version of a low-light photo to compare
against, so this scores the enhanced image against a handful of simple,
explainable heuristics rather than a learned metric. Each sub-score is
0-100 and the overall score is a weighted blend. The point isn't to be a
publishable IQA metric - it's to give the reflection loop something
concrete to act on ("still too dark" vs "noise still visible").
"""

from vision.exposure import brightness_stats
from vision.denoising import estimate_noise

TARGET_BRIGHTNESS_RANGE = (95, 175)
WEIGHTS = {"brightness": 0.35, "contrast": 0.25, "noise": 0.25, "clipping": 0.15}


def _brightness_score(stats: dict) -> float:
    lo, hi = TARGET_BRIGHTNESS_RANGE
    mean = stats["mean"]
    if lo <= mean <= hi:
        return 100.0
    distance = (lo - mean) if mean < lo else (mean - hi)
    return max(0.0, 100.0 - distance * 1.8)


def _contrast_score(stats: dict) -> float:
    # Flat, washed-out shadows (low std) is the classic low-light symptom.
    # A std below ~30 still reads as flat; above ~70 starts looking harsh.
    std = stats["std"]
    if 30 <= std <= 70:
        return 100.0
    distance = (30 - std) if std < 30 else (std - 70)
    return max(0.0, 100.0 - distance * 2.0)


def _noise_score(before_noise: float, after_noise: float) -> float:
    if before_noise <= 1e-6:
        return 100.0
    reduction = 1 - (after_noise / before_noise)
    # Reward noise reduction, but very aggressive smoothing (>85% drop in
    # Laplacian variance) usually means real detail got wiped too.
    if reduction < 0:
        return max(0.0, 50.0 + reduction * 100)
    if reduction > 0.85:
        overshoot = reduction - 0.85
        return max(0.0, 100.0 - overshoot * 300)
    return 50.0 + reduction * 58.8  # scales 0->0.85 reduction to 50->100


def _clipping_penalty(stats: dict) -> float:
    penalty = stats["percent_dark"] * 0.6 + stats["percent_clipped_bright"] * 1.2
    return max(0.0, 100.0 - penalty)


def score_image(before, after) -> dict:
    before_stats = brightness_stats(before)
    after_stats = brightness_stats(after)
    before_noise = estimate_noise(before)
    after_noise = estimate_noise(after)

    scores = {
        "brightness": round(_brightness_score(after_stats), 1),
        "contrast": round(_contrast_score(after_stats), 1),
        "noise": round(_noise_score(before_noise, after_noise), 1),
        "clipping": round(_clipping_penalty(after_stats), 1),
    }
    overall = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "scores": scores,
        "overall": round(overall, 1),
        "after_stats": after_stats,
        "before_noise": round(before_noise, 1),
        "after_noise": round(after_noise, 1),
    }

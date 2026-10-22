"""
Rule-based enhancement planner.

This is the fallback path: no API key, no network, or the LLM call fails
for any reason, and the pipeline still needs to produce a plan. It encodes
the same judgment calls as the notes in this folder, just as executable
thresholds instead of prose - deliberately duplicated rather than parsed
out of the markdown, since keeping the "for humans" notes and the
executable rules separate makes it obvious which one is currently
steering a given run.
"""


def _pick_gamma(mean_brightness: float) -> tuple[float, str]:
    if mean_brightness < 35:
        return 0.45, "severely underexposed (mean < 35), strong gamma lift"
    if mean_brightness < 60:
        return 0.6, "clearly underexposed (mean < 60)"
    if mean_brightness < 75:
        return 0.75, "moderately dark (mean < 75)"
    if mean_brightness < 95:
        return 0.88, "mildly dark (mean < 95), light gamma touch"
    return 1.0, "brightness already in range, no gamma change"


def _pick_clahe(std_dev: float) -> tuple[float, str]:
    if std_dev < 20:
        return 3.0, "very flat contrast (std < 20)"
    if std_dev < 30:
        return 2.5, "somewhat flat contrast (std < 30)"
    return 2.0, "contrast in a reasonable range, default clip"


def _pick_denoise(noise_estimate: float, gamma: float) -> tuple[float, str]:
    base = 6.0
    reason = "low measured noise"
    if noise_estimate > 400:
        base, reason = 14.0, "high measured noise"
    elif noise_estimate > 150:
        base, reason = 10.0, "moderate measured noise"

    if gamma <= 0.6:
        base += 3.0
        reason += ", bumped up since gamma lift will amplify shadow noise"

    return min(base, 20.0), reason


def _apply_feedback(plan: dict, feedback: str) -> dict:
    feedback = (feedback or "").lower()
    notes = []

    if "too dark" in feedback:
        plan["gamma"] = max(0.35, plan["gamma"] - 0.1)
        notes.append("previous pass still read too dark, dropping gamma further")
    if "clipped" in feedback or "too bright" in feedback:
        plan["gamma"] = min(1.0, plan["gamma"] + 0.1)
        notes.append("previous pass clipped highlights, easing gamma back")
    if "flat" in feedback or "low contrast" in feedback:
        plan["clahe_clip"] = min(4.0, plan["clahe_clip"] + 0.5)
        notes.append("previous pass still flat, raising CLAHE clip limit")
    if "noise" in feedback and "over-smoothed" not in feedback:
        plan["denoise_strength"] = min(22.0, plan["denoise_strength"] + 4)
        notes.append("previous pass still noisy, increasing denoise strength")
    if "over-smoothed" in feedback or "lost detail" in feedback or "waxy" in feedback:
        plan["denoise_strength"] = max(3.0, plan["denoise_strength"] - 4)
        plan["sharpen_amount"] = min(0.8, plan["sharpen_amount"] + 0.15)
        notes.append("previous pass over-smoothed, easing denoise and adding sharpening back")

    if notes:
        plan["reasoning"].append("reflection adjustments: " + "; ".join(notes))
    return plan


def rule_based_plan(scene, feedback: str = "") -> dict:
    """scene is duck-typed - anything with mean_brightness, std_dev, and
    noise_estimate attributes works, so this has no hard dependency on the
    agents package."""
    gamma, gamma_reason = _pick_gamma(scene.mean_brightness)
    clahe_clip, clahe_reason = _pick_clahe(scene.std_dev)
    denoise_strength, denoise_reason = _pick_denoise(scene.noise_estimate, gamma)

    sharpen_amount = 0.6 if denoise_strength <= 10 else max(0.25, 0.6 - 0.02 * (denoise_strength - 10))

    plan = {
        "gamma": gamma,
        "clahe_clip": clahe_clip,
        "clahe_tile": 8,
        "denoise_method": "nlmeans",
        "denoise_strength": round(denoise_strength, 1),
        "sharpen_amount": round(sharpen_amount, 2),
        "reasoning": [
            f"gamma={gamma}: {gamma_reason}",
            f"clahe_clip={clahe_clip}: {clahe_reason}",
            f"denoise_strength={round(denoise_strength, 1)}: {denoise_reason}",
        ],
    }

    if feedback:
        plan = _apply_feedback(plan, feedback)

    return plan

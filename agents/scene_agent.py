from agents.models import SceneReport
from toolbox.registry import ToolRegistry

NOISE_THRESHOLD = 150
FLAT_STD_THRESHOLD = 30
UNDEREXPOSED_MEAN_THRESHOLD = 95


class SceneAgent:
    """First stop in the pipeline: figure out what's actually wrong with
    the image before deciding what to do about it. Kept deliberately dumb
    (no decisions here, just measurements + a plain-English summary) so the
    planning agent is the only place enhancement judgment calls get made."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def analyze(self, image) -> SceneReport:
        stats = self.registry.execute("brightness_stats", image)
        noise = self.registry.execute("estimate_noise", image)

        is_underexposed = stats["mean"] < UNDEREXPOSED_MEAN_THRESHOLD
        is_flat = stats["std"] < FLAT_STD_THRESHOLD
        is_noisy = noise > NOISE_THRESHOLD

        summary = self._describe(stats, noise, is_underexposed, is_flat, is_noisy)

        return SceneReport(
            mean_brightness=stats["mean"],
            std_dev=stats["std"],
            percent_dark=stats["percent_dark"],
            percent_clipped_bright=stats["percent_clipped_bright"],
            noise_estimate=noise,
            is_underexposed=is_underexposed,
            is_noisy=is_noisy,
            is_flat=is_flat,
            summary=summary,
        )

    @staticmethod
    def _describe(stats, noise, is_underexposed, is_flat, is_noisy) -> str:
        mean = stats["mean"]
        if mean < 35:
            exposure_phrase = "severely underexposed"
        elif mean < 60:
            exposure_phrase = "clearly underexposed"
        elif mean < 95:
            exposure_phrase = "mildly underexposed"
        else:
            exposure_phrase = "reasonably exposed"

        extras = []
        if is_flat:
            extras.append("flat, low-contrast shadows")
        if is_noisy:
            extras.append("visible sensor noise")
        extra_phrase = f", with {' and '.join(extras)}" if extras else ""

        return (
            f"{exposure_phrase} image{extra_phrase}. "
            f"mean brightness {mean}, contrast std {stats['std']}, "
            f"noise estimate {round(noise, 1)}, "
            f"{stats['percent_dark']}% dark pixels, "
            f"{stats['percent_clipped_bright']}% blown highlights."
        )

"""Runs an EnhancementPlan against an image via the tool registry.

The denoise step specifically has a fallback chain: if the plan's
preferred method throws (or isn't available in the current OpenCV build),
we drop to the next one rather than letting the whole run die. Gaussian
blur is always last in the chain since it can't meaningfully fail on a
valid image array."""

from agents.models import EnhancementPlan
from toolbox.registry import ToolRegistry


class DenoiseAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def apply(self, image, plan: EnhancementPlan):
        """Returns (enhanced_image, denoise_method_used, fallback_triggered)."""
        working = image

        if plan.gamma < 0.99:
            working = self.registry.execute("apply_gamma", working, plan.gamma)

        if plan.clahe_clip > 0:
            working = self.registry.execute(
                "apply_clahe", working, clip_limit=plan.clahe_clip, tile_grid=plan.clahe_tile
            )

        working, method_used, fallback_triggered = self._denoise_with_fallback(working, plan)

        if plan.sharpen_amount > 0:
            working = self.registry.execute("unsharp_mask", working, amount=plan.sharpen_amount)

        return working, method_used, fallback_triggered

    def _denoise_with_fallback(self, image, plan: EnhancementPlan):
        chain = self._build_chain(plan.denoise_method, plan.denoise_strength)
        fallback_triggered = False
        last_error = None

        for tool_name, kwargs in chain:
            try:
                result = self.registry.execute(tool_name, image, **kwargs)
                return result, tool_name, fallback_triggered
            except Exception as exc:  # noqa: BLE001 - try the next method in the chain
                last_error = exc
                fallback_triggered = True

        # gaussian blur is always last in the chain and takes a plain array,
        # so in practice we should never get here on a valid image
        raise RuntimeError(f"every denoise method in the chain failed, last error: {last_error}")

    @staticmethod
    def _build_chain(preferred_method: str, strength: float):
        strength = max(3.0, min(strength, 22.0))
        nlmeans = ("denoise_nlmeans", {"strength": strength})
        bilateral = ("denoise_bilateral", {"diameter": 9, "sigma_color": 75, "sigma_space": 75})
        gaussian = ("denoise_gaussian", {"ksize": 5})

        if preferred_method == "bilateral":
            return [bilateral, nlmeans, gaussian]
        return [nlmeans, bilateral, gaussian]

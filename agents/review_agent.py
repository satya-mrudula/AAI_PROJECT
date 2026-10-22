import config
from agents.models import QualityReport
from vision.quality_metrics import score_image
from toolbox.registry import ToolRegistry

ISSUE_SCORE_THRESHOLD = 70.0


class ReviewAgent:
    def __init__(self, threshold: float = config.QUALITY_PASS_THRESHOLD, registry: ToolRegistry | None = None):
        self.threshold = threshold
        self.registry = registry

    def evaluate(self, before, after) -> QualityReport:
        result = self.registry.execute("evaluate_quality", before, after) if self.registry else score_image(before, after)
        passed = result["overall"] >= self.threshold
        feedback = self._build_feedback(result)
        return QualityReport(
            scores=result["scores"],
            overall=result["overall"],
            passed=passed,
            feedback=feedback,
        )

    @staticmethod
    def _build_feedback(result: dict) -> str:
        scores = result["scores"]
        after_stats = result["after_stats"]
        issues = []

        if scores["brightness"] < ISSUE_SCORE_THRESHOLD:
            issues.append("too dark" if after_stats["mean"] < 130 else "too bright / clipped")

        if scores["contrast"] < ISSUE_SCORE_THRESHOLD:
            issues.append("flat, low contrast")

        if scores["noise"] < ISSUE_SCORE_THRESHOLD:
            wiped_too_much = result["after_noise"] < result["before_noise"] * 0.15
            issues.append("over-smoothed, lost detail" if wiped_too_much else "noise still visible")

        if scores["clipping"] < ISSUE_SCORE_THRESHOLD:
            if after_stats["percent_clipped_bright"] > after_stats["percent_dark"]:
                issues.append("highlights clipped")
            else:
                issues.append("shadows crushed")

        return "; ".join(issues) if issues else "no major issues"

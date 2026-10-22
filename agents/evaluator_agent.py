"""Deterministic workflow-level evaluator for the image enhancement experiments.

The ReviewAgent evaluates one image attempt. This EvaluatorAgent evaluates the
whole experiment: task success, tool reliability, latency, quality vs baseline,
and reflection ablations.
"""
from evaluation.metrics import summarize


class EvaluatorAgent:
    def evaluate(self, records: list[dict], mcp_enabled: bool = False) -> dict:
        summary = summarize(records)
        summary["mcp_enabled"] = bool(mcp_enabled)
        summary["case_categories"] = {}
        for category in sorted({record.get("category", "uncategorized") for record in records}):
            group = [record for record in records if record.get("category", "uncategorized") == category]
            summary["case_categories"][category] = {
                "cases": len(group),
                "successful_cases": sum(1 for record in group if record.get("success")),
                "success_rate": sum(1 for record in group if record.get("success")) / len(group) if group else 0.0,
            }
        positive_records = [record for record in records if record.get("agentic_score") is not None]
        summary["reflection_ablation"] = {}
        for reflection_count in sorted({record.get("max_reflections", 0) for record in positive_records}):
            group = [record for record in positive_records if record.get("max_reflections", 0) == reflection_count]
            summary["reflection_ablation"][str(reflection_count)] = {
                "cases": len(group),
                "average_agentic_quality_score": sum(record["agentic_score"] for record in group) / len(group) if group else None,
                "average_latency_ms": sum(record.get("latency_ms", 0) for record in group) / len(group) if group else None,
                "average_baseline_quality_score": sum(record["baseline_score"] for record in group) / len(group) if group else None,
            }
        return summary

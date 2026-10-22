"""Metric helpers used by the reproducible evaluation suite."""
import math


def percentile(values, percent):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    index = (len(values) - 1) * percent / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def summarize(records):
    total = len(records)
    successful = [r for r in records if r.get("success")]
    latencies = [r.get("latency_ms", 0) for r in successful]
    tool_total = sum(len(r.get("tool_calls", [])) for r in successful)
    tool_success = sum(sum(1 for c in r.get("tool_calls", []) if c.get("ok")) for r in successful)
    return {
        "total_cases": total,
        "successful_cases": len(successful),
        "task_success_rate": len(successful) / total if total else 0.0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
        "tool_calls": tool_total,
        "tool_success_rate": tool_success / tool_total if tool_total else None,
        "average_agentic_quality_score": sum(r.get("agentic_score", 0) for r in successful) / len(successful) if successful else None,
        "average_baseline_quality_score": sum(r.get("baseline_score", 0) for r in successful) / len(successful) if successful else None,
        "estimated_cost_usd": sum(r.get("estimated_cost_usd") or 0 for r in successful),
        "cost_note": "Cost is zero for heuristic-only runs. LLM cost is reported only when token usage and configured per-million token rates are available.",
    }

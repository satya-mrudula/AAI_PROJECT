from evaluation.metrics import percentile, summarize


def test_percentile_handles_empty_and_single_values():
    assert percentile([], 95) == 0.0
    assert percentile([12], 95) == 12


def test_summary_reports_success_and_tool_metrics():
    result = summarize([
        {"success": True, "latency_ms": 10, "agentic_score": 80, "baseline_score": 70,
         "tool_calls": [{"ok": True}, {"ok": False}], "estimated_cost_usd": 0.01},
        {"success": False, "latency_ms": 0, "tool_calls": []},
    ])
    assert result["total_cases"] == 2
    assert result["task_success_rate"] == 0.5
    assert result["tool_success_rate"] == 0.5
    assert result["average_agentic_quality_score"] == 80


def test_evaluator_reports_category_and_reflection_ablation():
    from agents.evaluator_agent import EvaluatorAgent
    records = [
        {"test_id": "A_R0", "category": "normal", "success": True, "max_reflections": 0, "latency_ms": 10, "agentic_score": 70, "baseline_score": 65, "tool_calls": []},
        {"test_id": "A_R1", "category": "normal", "success": True, "max_reflections": 1, "latency_ms": 15, "agentic_score": 75, "baseline_score": 65, "tool_calls": []},
    ]
    summary = EvaluatorAgent().evaluate(records, mcp_enabled=False)
    assert summary["case_categories"]["normal"]["success_rate"] == 1.0
    assert summary["reflection_ablation"]["0"]["average_agentic_quality_score"] == 70
    assert summary["reflection_ablation"]["1"]["average_agentic_quality_score"] == 75


def test_cost_formula_uses_provider_reported_tokens():
    from evaluation.cost import calculate_cost
    assert abs(calculate_cost(4000, 1000, 1.0, 5.0) - 0.009) < 1e-12


def test_cost_formula_rejects_negative_inputs():
    from evaluation.cost import calculate_cost
    import pytest
    with pytest.raises(ValueError):
        calculate_cost(-1, 100, 1.0, 5.0)

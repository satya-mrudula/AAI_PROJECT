"""
Wires the agents together and owns the reflection loop:

  scene agent -> planning agent -> denoise agent -> review agent
                       ^                                  |
                       '------- feedback, retry -----------'

If nothing clears the quality bar within the allowed rounds, a fixed
conservative preset is applied as a last resort so the pipeline never
hands back something worse than a reasonable safe default.
"""

from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import time
import uuid

import config
from agents.denoise_agent import DenoiseAgent
from agents.models import Attempt, EnhancementPlan, RunReport
from agents.planning_agent import PlanningAgent
from agents.reporting import write_report
from evaluation.cost import calculate_cost
from agents.review_agent import ReviewAgent
from agents.scene_agent import SceneAgent
from knowledge.retriever import KnowledgeRetriever
from toolbox.image_tools import build_default_registry, load_image, save_image


class Pipeline:
    def __init__(self, max_reflection_rounds: int = config.MAX_REFLECTION_ROUNDS, use_mcp_tools: bool | None = None):
        self.max_reflection_rounds = max_reflection_rounds
        self.use_mcp_tools = config.USE_MCP_TOOLS if use_mcp_tools is None else use_mcp_tools
        self.registry = build_default_registry(use_mcp=self.use_mcp_tools)
        self.scene_agent = SceneAgent(self.registry)
        self.planning_agent = PlanningAgent(KnowledgeRetriever())
        self.denoise_agent = DenoiseAgent(self.registry)
        self.review_agent = ReviewAgent(registry=self.registry)

    def run(self, input_path: str, output_path: str = None, report_path: str = None) -> RunReport:
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        try:
            report = self._run_core(input_path, output_path=output_path, report_path=report_path)
        finally:
            close = getattr(self.registry, "close", None)
            if callable(close):
                close()
        report.run_id = run_id
        report.timestamp = datetime.now(timezone.utc).isoformat()
        report.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        report.tool_calls = [asdict(call) for call in self.registry.call_log]
        report.error_count = sum(1 for call in report.tool_calls if not call.get("ok", False))
        report.retry_count = max(0, len(report.attempts) - 1) + sum(1 for attempt in report.attempts if attempt.denoise_fallback_triggered)
        report.input_tokens = int(getattr(self.planning_agent, "input_tokens", 0))
        report.output_tokens = int(getattr(self.planning_agent, "output_tokens", 0))
        report.llm_call_attempts = int(getattr(self.planning_agent, "llm_call_attempts", 0))
        report.llm_failed_calls = int(getattr(self.planning_agent, "llm_failed_calls", 0))
        in_rate = config.GEMINI_INPUT_PRICE_PER_MILLION
        out_rate = config.GEMINI_OUTPUT_PRICE_PER_MILLION
        if report.llm_call_attempts == 0:
            report.estimated_cost_usd = 0.0
            report.cost_status = "no_llm_api_call_attempted"
        elif report.llm_failed_calls:
            report.estimated_cost_usd = None
            report.cost_status = "one_or_more_llm_calls_failed_and_provider_usage_is_unavailable"
        elif in_rate is not None and out_rate is not None:
            report.estimated_cost_usd = calculate_cost(report.input_tokens, report.output_tokens, float(in_rate), float(out_rate))
            report.cost_status = "estimated_from_reported_tokens_and_configured_rates"
        else:
            report.estimated_cost_usd = None
            report.cost_status = "gemini_token_usage_recorded_but_pricing_rates_not_configured"
        write_report(report, report.report_path)
        runs_dir = Path(config.RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)
        with (runs_dir / f"{run_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, indent=2, ensure_ascii=False)
        return report

    def _run_core(self, input_path: str, output_path: str = None, report_path: str = None) -> RunReport:
        original = load_image(input_path)
        scene = self.scene_agent.analyze(original)

        attempts: list[Attempt] = []
        best_image = None
        best_score = -1.0
        feedback = ""

        for round_number in range(1, self.max_reflection_rounds + 2):
            plan = self.planning_agent.decide(scene, feedback)
            enhanced, method_used, fallback_triggered = self.denoise_agent.apply(original, plan)
            quality = self.review_agent.evaluate(original, enhanced)

            attempts.append(
                Attempt(
                    round_number=round_number,
                    plan=plan,
                    quality=quality,
                    denoise_method_used=method_used,
                    denoise_fallback_triggered=fallback_triggered,
                )
            )

            if quality.overall > best_score:
                best_score, best_image = quality.overall, enhanced

            if quality.passed:
                break

            feedback = quality.feedback

        used_safety_preset = False
        if best_score < self.review_agent.threshold:
            best_image, best_score, used_safety_preset = self._apply_safety_preset(original, attempts)

        output_path = output_path or self._default_output_path(input_path)
        report_path = report_path or str(Path(output_path).with_suffix("")) + "_report.md"

        save_image(best_image, output_path)

        run_report = RunReport(
            input_path=str(input_path),
            output_path=str(output_path),
            report_path=str(report_path),
            scene=scene,
            attempts=attempts,
            used_safety_preset=used_safety_preset,
            final_overall_score=best_score,
        )
        write_report(run_report, report_path)
        return run_report

    def _apply_safety_preset(self, original, attempts: list[Attempt]):
        plan = EnhancementPlan(
            **config.SAFETY_PRESET,
            reasoning=["no attempt cleared the quality threshold, applying the conservative safety preset"],
            source="safety-preset",
        )
        image, method_used, fallback_triggered = self.denoise_agent.apply(original, plan)
        quality = self.review_agent.evaluate(original, image)
        attempts.append(
            Attempt(
                round_number=len(attempts) + 1,
                plan=plan,
                quality=quality,
                denoise_method_used=method_used,
                denoise_fallback_triggered=fallback_triggered,
            )
        )
        return image, quality.overall, True

    @staticmethod
    def _default_output_path(input_path: str) -> str:
        stem = Path(input_path).stem
        return str(Path("outputs") / f"{stem}_enhanced.jpg")

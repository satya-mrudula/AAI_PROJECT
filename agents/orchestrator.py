"""
Coordinates the multi-agent image enhancement pipeline.

Flow:
Scene Agent -> Planning Agent -> Denoise Agent -> Review Agent
                         ^                         |
                         |------ Feedback ----------|

Keeps the configured reflection rounds and applies a conservative
safety preset if no attempt passes the quality threshold.
"""

from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import json
import time
import uuid

import config

from agents.denoise_agent import DenoiseAgent
from agents.models import Attempt, EnhancementPlan, RunReport
from agents.planning_agent import PlanningAgent
from agents.reporting import write_report
from agents.review_agent import ReviewAgent
from agents.scene_agent import SceneAgent

from evaluation.cost import calculate_cost
from knowledge.retriever import KnowledgeRetriever
from toolbox.image_tools import (
    build_default_registry,
    load_image,
    save_image,
)


class Pipeline:

    def __init__(
        self,
        max_reflection_rounds: int = config.MAX_REFLECTION_ROUNDS,
        use_mcp_tools: bool | None = None,
    ):
        self.max_reflection_rounds = max(0, int(max_reflection_rounds))

        self.use_mcp_tools = (
            config.USE_MCP_TOOLS
            if use_mcp_tools is None
            else use_mcp_tools
        )

        # Local tools are always available as a fallback.
        self.local_registry = build_default_registry(use_mcp=False)

        self.registry = self.local_registry
        self.mcp_error = None

        if self.use_mcp_tools:
            try:
                self.registry = build_default_registry(use_mcp=True)
            except Exception as exc:
                self.mcp_error = str(exc)
                self.registry = self.local_registry

        self.scene_agent = SceneAgent(self.registry)
        self.planning_agent = PlanningAgent(KnowledgeRetriever())
        self.denoise_agent = DenoiseAgent(self.registry)
        self.review_agent = ReviewAgent(registry=self.registry)

        # Independent local agents for MCP failure recovery.
        self.local_scene_agent = SceneAgent(self.local_registry)
        self.local_denoise_agent = DenoiseAgent(self.local_registry)
        self.local_review_agent = ReviewAgent(
            registry=self.local_registry
        )

    def run(
        self,
        input_path: str,
        output_path: str = None,
        report_path: str = None,
    ) -> RunReport:

        started = time.perf_counter()
        run_id = str(uuid.uuid4())

        report = self._run_core(
            input_path,
            output_path=output_path,
            report_path=report_path,
        )

        report.run_id = run_id
        report.timestamp = datetime.now(timezone.utc).isoformat()

        report.latency_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )

        # Preserve logs from both MCP and local execution.
        calls = (
            list(self.registry.call_log)
            if self.registry is not self.local_registry
            else []
        )

        calls.extend(self.local_registry.call_log)

        report.tool_calls = [
            asdict(call) for call in calls
        ]

        report.error_count = sum(
            1 for call in report.tool_calls
            if not call.get("ok", False)
        )

        report.retry_count = (
            max(0, len(report.attempts) - 1)
            + sum(
                1 for attempt in report.attempts
                if attempt.denoise_fallback_triggered
            )
        )

        report.input_tokens = int(
            getattr(self.planning_agent, "input_tokens", 0)
        )

        report.output_tokens = int(
            getattr(self.planning_agent, "output_tokens", 0)
        )

        report.llm_call_attempts = int(
            getattr(self.planning_agent, "llm_call_attempts", 0)
        )

        report.llm_failed_calls = int(
            getattr(self.planning_agent, "llm_failed_calls", 0)
        )

        input_rate = config.GEMINI_INPUT_PRICE_PER_MILLION
        output_rate = config.GEMINI_OUTPUT_PRICE_PER_MILLION

        if report.llm_call_attempts == 0:
            report.estimated_cost_usd = 0.0
            report.cost_status = "no_llm_api_call_attempted"

        elif report.llm_failed_calls:
            report.estimated_cost_usd = None
            report.cost_status = (
                "one_or_more_llm_calls_failed_and_provider_usage_is_unavailable"
            )

        elif input_rate is not None and output_rate is not None:
            report.estimated_cost_usd = calculate_cost(
                report.input_tokens,
                report.output_tokens,
                float(input_rate),
                float(output_rate),
            )

            report.cost_status = (
                "estimated_from_reported_tokens_and_configured_rates"
            )

        else:
            report.estimated_cost_usd = None
            report.cost_status = (
                "gemini_token_usage_recorded_but_pricing_rates_not_configured"
            )

        write_report(report, report.report_path)

        runs_dir = Path(config.RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)

        with (runs_dir / f"{run_id}.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                asdict(report),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        return report

    def _run_core(
        self,
        input_path: str,
        output_path: str = None,
        report_path: str = None,
    ) -> RunReport:

        original = load_image(input_path)
        original_shape = original.shape

        # Step 1: Analyze the input image.
        try:
            scene = self.scene_agent.analyze(original)

        except Exception:
            if self.registry is self.local_registry:
                raise

            # MCP analysis failed. Retry using local tools.
            scene = self.local_scene_agent.analyze(original)

        attempts = []

        best_image = original.copy()
        best_score = -1.0
        feedback = ""

        # Initial attempt + configured reflection rounds.
        for round_number in range(
            1,
            self.max_reflection_rounds + 2,
        ):

            # Step 2: Generate an enhancement plan.
            plan = self.planning_agent.decide(
                scene,
                feedback,
            )

            # Step 3: Apply enhancement.
            try:
                enhanced, method_used, fallback_triggered = (
                    self.denoise_agent.apply(original, plan)
                )

            except Exception:
                if self.registry is self.local_registry:
                    raise

                # MCP processing failed. Retry locally.
                enhanced, method_used, fallback_triggered = (
                    self.local_denoise_agent.apply(original, plan)
                )

                fallback_triggered = True

            self._validate_image(
                enhanced,
                original_shape,
            )

            # Step 4: Evaluate image quality.
            try:
                quality = self.review_agent.evaluate(
                    original,
                    enhanced,
                )

            except Exception:
                if self.registry is self.local_registry:
                    raise

                # MCP quality evaluation failed.
                quality = self.local_review_agent.evaluate(
                    original,
                    enhanced,
                )

            attempts.append(
                Attempt(
                    round_number=round_number,
                    plan=plan,
                    quality=quality,
                    denoise_method_used=method_used,
                    denoise_fallback_triggered=fallback_triggered,
                )
            )

            # Keep the best result seen so far.
            if quality.overall > best_score:
                best_score = quality.overall
                best_image = enhanced.copy()

            # Stop when the quality threshold is reached.
            if quality.passed:
                break

            # Send review feedback into the next planning round.
            feedback = quality.feedback

        # Step 5: Apply safety preset if necessary.
        used_safety_preset = False

        if best_score < self.review_agent.threshold:

            best_image, best_score, used_safety_preset = (
                self._apply_safety_preset(
                    original,
                    original_shape,
                    attempts,
                )
            )

        # Step 6: Save final output.
        output_path = (
            output_path
            or self._default_output_path(input_path)
        )

        report_path = (
            report_path
            or str(Path(output_path).with_suffix(""))
            + "_report.md"
        )

        save_image(best_image, output_path)

        return RunReport(
            input_path=str(input_path),
            output_path=str(output_path),
            report_path=str(report_path),
            scene=scene,
            attempts=attempts,
            used_safety_preset=used_safety_preset,
            final_overall_score=best_score,
        )

    def _apply_safety_preset(
        self,
        original,
        original_shape,
        attempts,
    ):

        plan = EnhancementPlan(
            **config.SAFETY_PRESET,
            reasoning=[
                "No attempt cleared the quality threshold. "
                "Applying the conservative safety preset."
            ],
            source="safety-preset",
        )

        try:
            image, method_used, fallback_triggered = (
                self.denoise_agent.apply(original, plan)
            )

        except Exception:
            if self.registry is self.local_registry:
                raise

            image, method_used, fallback_triggered = (
                self.local_denoise_agent.apply(original, plan)
            )

            fallback_triggered = True

        self._validate_image(image, original_shape)

        try:
            quality = self.review_agent.evaluate(
                original,
                image,
            )

        except Exception:
            if self.registry is self.local_registry:
                raise

            quality = self.local_review_agent.evaluate(
                original,
                image,
            )

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
    def _validate_image(image, original_shape):

        if image is None:
            raise ValueError(
                "Enhancement returned an empty image."
            )

        if image.shape != original_shape:
            raise ValueError(
                "Enhancement changed the original image dimensions "
                f"from {original_shape} to {image.shape}."
            )

    @staticmethod
    def _default_output_path(input_path: str) -> str:

        stem = Path(input_path).stem

        return str(
            Path("outputs") / f"{stem}_enhanced.jpg"
        )

    def close(self):
        """Close the MCP session when the pipeline is no longer needed."""

        if self.registry is not self.local_registry:
            close = getattr(self.registry, "close", None)

            if callable(close):
                close()

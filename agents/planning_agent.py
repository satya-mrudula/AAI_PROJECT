"""Image enhancement planning using Gemini + RAG, with a heuristic fallback.

If GEMINI_API_KEY is configured, Gemini selects enhancement parameters from
scene statistics and retrieved notes. Any API or parsing failure falls back
to the deterministic local planner so image enhancement remains available.
"""

import json
import re

import config
from agents.models import EnhancementPlan, SceneReport
from knowledge.heuristics import rule_based_plan
from knowledge.retriever import KnowledgeRetriever

try:
    from google import genai
except ImportError:
    genai = None


PLAN_FIELDS = ("gamma", "clahe_clip", "denoise_strength", "sharpen_amount")

SYSTEM_PROMPT = (
    "You are the planning stage of an image enhancement pipeline. Given "
    "measurements of a low-light photo and notes retrieved from a small "
    "internal knowledge base, choose enhancement parameters. Reply with "
    "ONLY a JSON object, no prose, no markdown fences. Fields: "
    "gamma (float, 0.3-1.0), clahe_clip (float, 1.5-4.0), clahe_tile (int, "
    "usually 8), denoise_method (\"nlmeans\" or \"bilateral\"), "
    "denoise_strength (float, 3-22), sharpen_amount (float, 0.2-0.8), "
    "reasoning (one short sentence)."
)


class PlanningAgent:
    def __init__(self, retriever: KnowledgeRetriever):
        self.retriever = retriever
        self.last_llm_error = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.llm_call_attempts = 0
        self.llm_failed_calls = 0
        self._client = self._init_client()

    @staticmethod
    def _init_client():
        if not config.GEMINI_API_KEY or genai is None:
            return None
        try:
            return genai.Client(api_key=config.GEMINI_API_KEY)
        except Exception:
            return None

    def decide(self, scene: SceneReport, feedback: str = "") -> EnhancementPlan:
        if self._client is not None:
            query = scene.summary if not feedback else f"{scene.summary} Prior attempt feedback: {feedback}"
            retrieved = self.retriever.retrieve(query, k=config.RAG_TOP_K)
            plan = self._decide_with_llm(scene, feedback, retrieved)
            if plan is not None:
                return plan
        return self._decide_with_heuristic(scene, feedback)

    def _decide_with_llm(self, scene, feedback, retrieved):
        notes_block = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk, _score in retrieved) or "(no notes matched)"
        user_prompt = (
            f"Scene measurements:\n"
            f"- mean brightness: {scene.mean_brightness}\n"
            f"- contrast (std dev): {scene.std_dev}\n"
            f"- percent dark pixels: {scene.percent_dark}\n"
            f"- percent blown highlights: {scene.percent_clipped_bright}\n"
            f"- noise estimate: {scene.noise_estimate}\n\n"
            f"Retrieved notes:\n{notes_block}\n\n"
            f"Prior attempt feedback: {feedback or '(none, this is the first pass)'}\n\n"
            "Return the JSON plan now."
        )

        try:
            self.llm_call_attempts += 1
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": 400,
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self.input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
                self.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
            self.llm_calls += 1
            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise ValueError("Gemini returned an empty response")
            data = json.loads(self._strip_fences(raw_text))

            for key in PLAN_FIELDS:
                if key not in data:
                    raise ValueError(f"model response missing '{key}'")

            gamma = float(data["gamma"])
            clahe_clip = float(data["clahe_clip"])
            denoise_strength = float(data["denoise_strength"])
            sharpen_amount = float(data["sharpen_amount"])
            if not 0.3 <= gamma <= 1.0:
                raise ValueError("Gemini gamma is outside the allowed range")
            if not 1.5 <= clahe_clip <= 4.0:
                raise ValueError("Gemini clahe_clip is outside the allowed range")
            if not 3.0 <= denoise_strength <= 22.0:
                raise ValueError("Gemini denoise_strength is outside the allowed range")
            if not 0.2 <= sharpen_amount <= 0.8:
                raise ValueError("Gemini sharpen_amount is outside the allowed range")
            denoise_method = data.get("denoise_method", "nlmeans")
            if denoise_method not in {"nlmeans", "bilateral"}:
                raise ValueError("Gemini returned an unsupported denoise_method")

            return EnhancementPlan(
                gamma=gamma,
                clahe_clip=clahe_clip,
                clahe_tile=int(data.get("clahe_tile", 8)),
                denoise_method=denoise_method,
                denoise_strength=denoise_strength,
                sharpen_amount=sharpen_amount,
                reasoning=[str(data.get("reasoning", "")).strip() or "Gemini decision"],
                source="gemini+rag",
                knowledge_used=[chunk.chunk_id for chunk, _score in retrieved],
            )
        except Exception as exc:  # network, auth, malformed output, etc. -> fallback
            self.llm_failed_calls += 1
            self.last_llm_error = f"{type(exc).__name__}: {exc}"
            return None

    def _decide_with_heuristic(self, scene, feedback) -> EnhancementPlan:
        raw = rule_based_plan(scene, feedback)
        return EnhancementPlan(
            gamma=raw["gamma"],
            clahe_clip=raw["clahe_clip"],
            clahe_tile=raw["clahe_tile"],
            denoise_method=raw["denoise_method"],
            denoise_strength=raw["denoise_strength"],
            sharpen_amount=raw["sharpen_amount"],
            reasoning=raw["reasoning"],
            source="heuristic-fallback",
            knowledge_used=[],
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        return match.group(1) if match else text

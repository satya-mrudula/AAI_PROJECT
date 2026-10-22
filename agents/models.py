"""Plain dataclasses passed between agents. Keeping these separate from the
agent classes themselves so nothing has to import an agent just to read the
shape of the data it produces."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class SceneReport:
    mean_brightness: float
    std_dev: float
    percent_dark: float
    percent_clipped_bright: float
    noise_estimate: float
    is_underexposed: bool
    is_noisy: bool
    is_flat: bool
    summary: str  # natural-language description, doubles as the RAG query


@dataclass
class EnhancementPlan:
    gamma: float
    clahe_clip: float
    clahe_tile: int
    denoise_method: str
    denoise_strength: float
    sharpen_amount: float
    reasoning: list[str]
    source: str  # "llm+rag" or "heuristic-fallback"
    knowledge_used: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    scores: dict
    overall: float
    passed: bool
    feedback: str


@dataclass
class Attempt:
    round_number: int
    plan: EnhancementPlan
    quality: QualityReport
    denoise_method_used: str
    denoise_fallback_triggered: bool


@dataclass
class RunReport:
    input_path: str
    output_path: str
    report_path: str
    scene: SceneReport
    attempts: list[Attempt]
    used_safety_preset: bool
    final_overall_score: float
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_call_attempts: int = 0
    llm_failed_calls: int = 0
    estimated_cost_usd: float | None = None
    cost_status: str = "not_available"

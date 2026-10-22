"""Project-wide constants. Nothing here should need touching per-run - use
CLI flags in enhance.py for that. This is for the knobs that define
"what does good enough look like" and "how far will we go to get there"."""

import os

from dotenv import load_dotenv

load_dotenv()

# minimum overall quality score (0-100) before the reflection loop stops
QUALITY_PASS_THRESHOLD = 72.0

# extra attempts allowed beyond the first pass
MAX_REFLECTION_ROUNDS = 2

# how many knowledge chunks get pulled into the LLM planner's context
RAG_TOP_K = 3

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# applied as a last resort if every reflection round still fails to pass -
# deliberately conservative so it can't make things worse than the original
SAFETY_PRESET = {
    "gamma": 0.85,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "denoise_method": "bilateral",
    "denoise_strength": 8.0,
    "sharpen_amount": 0.3,
}

# Set LOWLIGHT_USE_MCP=1 to route image operations through the MCP server.
USE_MCP_TOOLS = os.environ.get("LOWLIGHT_USE_MCP", "0").strip().lower() in {"1", "true", "yes", "on"}
RUNS_DIR = os.environ.get("LOWLIGHT_RUNS_DIR", "runs")
GEMINI_INPUT_PRICE_PER_MILLION = os.environ.get("GEMINI_INPUT_PRICE_PER_MILLION")
GEMINI_OUTPUT_PRICE_PER_MILLION = os.environ.get("GEMINI_OUTPUT_PRICE_PER_MILLION")

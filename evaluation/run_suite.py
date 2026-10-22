"""Run the 30-case image evaluation suite and write JSON/CSV results.

Examples:
  python -m evaluation.run_suite
  LOWLIGHT_USE_MCP=1 python -m evaluation.run_suite --mcp
"""
import argparse
import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from agents.orchestrator import Pipeline
from agents.evaluator_agent import EvaluatorAgent
from evaluation.metrics import summarize
from toolbox.image_tools import load_image, save_image
from vision.exposure import apply_gamma
from vision.denoising import denoise_nlmeans
from vision.quality_metrics import score_image

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = Path(__file__).with_name("test_cases.json")


def create_variant(source_path: Path, transform: str, target_path: Path):
    image = load_image(str(source_path))
    max_side = max(image.shape[:2])
    if max_side > 256:
        scale = 256 / max_side
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if transform == "original":
        variant = image
    elif transform.startswith("darken_"):
        factor = float(transform.split("_", 1)[1])
        variant = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif transform.startswith("noise_"):
        sigma = float(transform.split("_", 1)[1])
        rng = np.random.default_rng(2026 + int(sigma))
        noise = rng.normal(0, sigma, image.shape)
        variant = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif transform == "low_contrast":
        mean = image.mean(axis=(0, 1), keepdims=True)
        variant = np.clip((image.astype(np.float32) - mean) * 0.45 + mean, 0, 255).astype(np.uint8)
    elif transform == "blur":
        variant = cv2.GaussianBlur(image, (7, 7), 1.8)
    elif transform == "resize_small":
        small = cv2.resize(image, (max(16, image.shape[1] // 3), max(16, image.shape[0] // 3)))
        variant = cv2.resize(small, (image.shape[1], image.shape[0]))
    elif transform == "gamma_bright":
        variant = apply_gamma(image, 1.4)
    elif transform == "mixed_noise_dark":
        rng = np.random.default_rng(707)
        dark = image.astype(np.float32) * 0.45
        variant = np.clip(dark + rng.normal(0, 12, image.shape), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown transform: {transform}")
    save_image(variant, str(target_path))


def run_evaluation_suite(use_mcp=False):
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    records = []
    with tempfile.TemporaryDirectory(prefix="lowlight-eval-") as temp:
        temp_dir = Path(temp)
        for case in cases:
            record = {"test_id": case["test_id"], "category": case["category"], "success": False, "max_reflections": case.get("max_reflections", 0), "notes": ""}
            input_path = temp_dir / f"{case['test_id']}_input.png"
            try:
                if case["expected"] == "controlled_input_error":
                    if case["input_kind"] == "missing_file":
                        candidate = temp_dir / "this_file_does_not_exist.png"
                    else:
                        candidate = temp_dir / f"{case['test_id']}.png"
                        candidate.write_bytes(b"not a valid image")
                    try:
                        Pipeline(max_reflection_rounds=0, use_mcp_tools=use_mcp).run(str(candidate), output_path=str(temp_dir / "unexpected.png"))
                    except (FileNotFoundError, ValueError) as exc:
                        record["success"] = True
                        record["notes"] = f"controlled input failure: {type(exc).__name__}"
                    else:
                        record["notes"] = "expected invalid input to be rejected"
                    records.append(record)
                    continue

                source = ROOT / case["source"]
                create_variant(source, case["transform"], input_path)
                original = load_image(str(input_path))
                baseline = denoise_nlmeans(apply_gamma(original, 0.72), strength=7.0)
                baseline_score = float(score_image(original, baseline)["overall"])
                for reflection_count in (0, 1):
                    case_record = {"test_id": f"{case['test_id']}_R{reflection_count}", "case_id": case["test_id"], "category": case["category"], "success": False, "max_reflections": reflection_count, "notes": ""}
                    output_path = temp_dir / f"{case['test_id']}_R{reflection_count}_enhanced.png"
                    try:
                        report = Pipeline(max_reflection_rounds=reflection_count, use_mcp_tools=use_mcp).run(
                            str(input_path), output_path=str(output_path), report_path=str(temp_dir / f"{case['test_id']}_R{reflection_count}.md")
                        )
                        case_record.update({
                            "success": output_path.exists(),
                            "latency_ms": report.latency_ms,
                            "agentic_score": report.final_overall_score,
                            "baseline_score": baseline_score,
                            "input_tokens": report.input_tokens,
                            "output_tokens": report.output_tokens,
                            "estimated_cost_usd": report.estimated_cost_usd,
                            "tool_calls": report.tool_calls,
                            "run_id": report.run_id,
                            "actual_rounds": len(report.attempts),
                            "notes": "same transformed input evaluated with the specified reflection limit; compared with fixed gamma+NL-means baseline",
                        })
                    except Exception as exc:
                        case_record["notes"] = f"ERROR: {type(exc).__name__}: {exc}"
                    records.append(case_record)
            except Exception as exc:
                record["notes"] = f"ERROR: {type(exc).__name__}: {exc}"
                records.append(record)

    summary = EvaluatorAgent().evaluate(records, mcp_enabled=use_mcp)
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    summary["unique_scenarios"] = len(cases)
    summary["baseline_definition"] = "Fixed gamma correction (gamma=0.72) followed by non-local means denoising (strength=7)."
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "evaluation_results.json").write_text(json.dumps({"summary": summary, "cases": records}, indent=2), encoding="utf-8")
    with (results_dir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            clean = dict(row)
            clean["tool_calls"] = json.dumps(clean.get("tool_calls", []))
            writer.writerow(clean)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp", action="store_true", help="send image operations through the MCP server")
    args = parser.parse_args()
    summary = run_evaluation_suite(use_mcp=args.mcp)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

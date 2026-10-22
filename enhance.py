#!/usr/bin/env python3
"""Command-line entry point.

    python enhance.py samples/dim_room.jpg
    python enhance.py path/to/photo.jpg --output outputs/photo_fixed.jpg --max-reflections 3
"""

import argparse
import sys

import config
from agents.orchestrator import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic low-light image enhancement")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--output", help="where to save the enhanced image (default: outputs/<name>_enhanced.jpg)")
    parser.add_argument("--report", help="where to save the markdown run report (default: next to the output image)")
    parser.add_argument(
        "--mcp", action=argparse.BooleanOptionalAction, default=None,
        help="use the MCP server for image tools (--no-mcp keeps local tools)",
    )
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=config.MAX_REFLECTION_ROUNDS,
        help="extra attempts allowed if the first pass doesn't clear the quality bar",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_mcp = config.USE_MCP_TOOLS if args.mcp is None else args.mcp
    pipeline = Pipeline(max_reflection_rounds=args.max_reflections, use_mcp_tools=use_mcp)

    try:
        report = pipeline.run(args.image, output_path=args.output, report_path=args.report)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"scene: {report.scene.summary}\n")
    for attempt in report.attempts:
        status = "passed" if attempt.quality.passed else "below threshold"
        print(f"round {attempt.round_number} [{attempt.plan.source}]: overall {attempt.quality.overall} ({status})")

    print()
    if report.used_safety_preset:
        print("no attempt cleared the quality bar - fell back to the conservative safety preset")
    print(f"saved enhanced image to {report.output_path}")
    print(f"saved run report to {report.report_path}")
    print(f"final quality score: {report.final_overall_score}")
    print(f"run ID: {report.run_id}")
    print(f"latency: {report.latency_ms} ms")
    print(f"tool calls: {len(report.tool_calls)}")
    print(f"estimated API cost: {report.estimated_cost_usd} USD ({report.cost_status})")


if __name__ == "__main__":
    main()

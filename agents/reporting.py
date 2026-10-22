from agents.models import RunReport


def render_markdown(report: RunReport) -> str:
    lines = [
        f"# Enhancement report - {report.input_path}",
        "",
        "## Scene",
        f"- {report.scene.summary}",
        "",
        "## Attempts",
    ]

    for attempt in report.attempts:
        plan = attempt.plan
        quality = attempt.quality
        status = "PASSED" if quality.passed else "did not meet threshold"

        lines.append(f"### Round {attempt.round_number} ({plan.source}) - {status}")
        lines.append(
            f"- plan: gamma={plan.gamma}, clahe_clip={plan.clahe_clip}, "
            f"denoise={attempt.denoise_method_used} (strength {plan.denoise_strength}), "
            f"sharpen={plan.sharpen_amount}"
        )
        if attempt.denoise_fallback_triggered:
            lines.append("- note: preferred denoise method failed, fallback chain kicked in")
        if plan.knowledge_used:
            lines.append(f"- knowledge used: {', '.join(plan.knowledge_used)}")
        for line in plan.reasoning:
            lines.append(f"  - {line}")
        lines.append(
            f"- quality: overall {quality.overall} "
            f"(brightness {quality.scores['brightness']}, contrast {quality.scores['contrast']}, "
            f"noise {quality.scores['noise']}, clipping {quality.scores['clipping']})"
        )
        if quality.feedback and quality.feedback != "no major issues":
            lines.append(f"- feedback: {quality.feedback}")
        lines.append("")

    lines.append("## Result")
    lines.append(f"- output: {report.output_path}")
    lines.append(f"- final score: {report.final_overall_score}")
    if report.used_safety_preset:
        lines.append("- no attempt cleared the quality threshold, so the conservative safety preset was used for the final output")

    return "\n".join(lines)


def write_report(report: RunReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

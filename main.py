"""FastAPI wrapper around the agent pipeline.

    uvicorn main:app --reload --port 8000
"""
from pathlib import Path
import json
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from enhance import Pipeline


app = FastAPI(title="Lowlight Agents API")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mcp_enabled": config.USE_MCP_TOOLS, "llm_provider": "google-gemini", "gemini_configured": bool(config.GEMINI_API_KEY)}


@app.get("/metrics/{run_id}")
def run_metrics(run_id: str):
    try:
        run_id = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="run_id must be a UUID") from exc
    path = Path(config.RUNS_DIR) / f"{run_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="run metrics not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/evaluation/run")
def run_evaluation():
    try:
        from evaluation.run_suite import run_evaluation_suite
        return run_evaluation_suite(use_mcp=config.USE_MCP_TOOLS)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"evaluation failed: {type(exc).__name__}: {exc}") from exc


@app.get("/evaluation/results")
def evaluation_results():
    path = Path("results/evaluation_results.json")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="evaluation has not been run yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/query")
@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    max_reflections: int = Query(config.MAX_REFLECTION_ROUNDS, ge=0, le=5),
):
    suffix = Path(file.filename or "input.jpg").suffix or ".jpg"

    work_dir = Path(tempfile.mkdtemp())
    input_path = work_dir / f"input{suffix}"
    output_path = work_dir / f"enhanced{suffix}"
    report_path = work_dir / "report.md"

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline = Pipeline(max_reflection_rounds=max_reflections, use_mcp_tools=config.USE_MCP_TOOLS)
        report = pipeline.run(
            str(input_path),
            output_path=str(output_path),
            report_path=str(report_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=f"image tool service unavailable: {exc}") from exc
    except Exception as exc:
        status_code = 503 if config.USE_MCP_TOOLS else 500
        raise HTTPException(status_code=status_code, detail=f"image enhancement failed: {type(exc).__name__}: {exc}") from exc

    return FileResponse(
        report.output_path,
        media_type="image/jpeg",
        filename=f"enhanced{suffix}",
        headers={
            "X-Final-Score": str(report.final_overall_score),
            "X-Used-Safety-Preset": str(report.used_safety_preset),
            "X-Run-ID": report.run_id,
            "X-Latency-MS": str(report.latency_ms),
        },
    )

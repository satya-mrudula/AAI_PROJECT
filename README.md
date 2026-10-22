# Agentic AI-Based Adaptive Low-Light Image Enhancement and Denoising

This project improves underexposed and noisy images through a multi-agent workflow. The existing image-processing pipeline is retained and extended with a real Model Context Protocol (MCP) server/client boundary, per-run telemetry, a separate evaluation suite, and API endpoints for run metrics and evaluation results.

## What changed in this version

- **Real MCP server/client:** `mcp_server/server.py` exposes deterministic image tools; `mcp_client/client.py` connects to it over stdio. The orchestration layer remains in the application, not inside the MCP server.
- **Agent workflow:** `SceneAgent → PlanningAgent → DenoiseAgent → ReviewAgent`, with reflection and a conservative safety preset.
- **Shared structured run data:** every successful run receives a `run_id`, UTC timestamp, latency, tool-call records, errors/retries, token usage, and cost status.
- **Separate evaluation suite:** 42 unique cases cover ordinary samples, synthetic degradations, stress variants, and invalid/corrupt inputs. Each of the 36 valid-image scenarios is run with reflection limits 0 and 1, so the suite records 78 executions and supports a paired reflection ablation. The runner compares the agentic pipeline with a fixed gamma-plus-NL-means baseline and writes JSON/CSV results.
- **API metrics:** `/health`, `/metrics/{run_id}`, `/evaluation/run`, and `/evaluation/results` are available alongside the existing `/enhance` endpoint.
- **Failure checks:** MCP tool input validation and metric helper tests are included. The MCP server is designed to return controlled tool errors rather than inventing image results.

> The included `results/evaluation_results.json` and `.csv` are results from a local-tool run in this environment (`mcp_enabled: false`); they are not presented as MCP-mode results. Run the evaluation suite on your machine in MCP mode and submit the generated results. The fixed baseline is deliberately simple and should be described as a baseline, not as a state-of-the-art model.

## Architecture

```text
User / Browser / CLI
        |
        v
   FastAPI or CLI
        |
        v
 Agent Orchestrator  ---- shared run state / run_id / telemetry
   |       |       |
   v       v       v
 Scene   Planner  Denoise -------- Review / Evaluator
 Agent   Agent    Agent                  |
   |       |       |                     |
   +-------+-------+---------------------+
                   |
                   v
              MCP Client
                   |
          stdio / MCP protocol
                   |
                   v
              MCP Server
                   |
     brightness, noise, gamma, CLAHE,
     denoise, sharpen, quality metrics
```

MCP is the capability boundary; it is not the multi-agent orchestration framework. The MCP server exposes deterministic tools, while the application decides which agent runs next and whether to reflect/retry.

## Agents

| Agent | Responsibility | Output |
|---|---|---|
| Scene Agent | Measures brightness, contrast, clipping and estimated noise | `SceneReport` |
| Planning Agent | Chooses enhancement parameters using Gemini + retrieved notes when configured, otherwise heuristic rules | `EnhancementPlan` |
| Denoise Agent | Applies gamma, CLAHE, denoising fallback chain and sharpening | Enhanced image + method/fallback status |
| Review Agent | Scores brightness, contrast, noise and clipping; supplies feedback for another reflection round | `QualityReport` |
| Evaluator Agent | Aggregates success, tool reliability, latency, quality vs baseline and reflection ablation across the suite | JSON and CSV result files |

## Project layout

```text
agents/                 Agent roles, shared models, orchestration and reports
mcp_server/              MCP server and deterministic tool implementations
mcp_client/              Persistent stdio MCP client adapter
vision/                  Image enhancement and quality functions
toolbox/                 Local registry and MCP-backed registry
knowledge/               Heuristics and retrieval notes
evaluation/               42-case evaluation dataset, metrics and runner
tests/                   Unit and contract tests
samples/                 Example input images
outputs/                 Example outputs and reports
runs/                    Per-run JSON telemetry (created at runtime)
results/                 Evaluation JSON/CSV output (created at runtime)
main.py                   FastAPI app
enhance.py                CLI entry point
Dockerfile                Container configuration
```

## Requirements

- Python 3.11 or 3.12 recommended
- Git
- Optional: Docker Desktop
- Optional: `GEMINI_API_KEY` for Gemini-backed planning. Without it, the heuristic planner runs locally.
- MCP SDK is installed from `requirements.txt`.

## Setup on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

The app loads values from your local `.env` file (created from `.env.example`) and also accepts environment variables from the shell. If you want MCP enabled, set the variable before running:

```bash
export LOWLIGHT_USE_MCP=1
```

To use local tools instead of MCP for debugging:

```bash
export LOWLIGHT_USE_MCP=0
```

The MCP client launches the local server as a subprocess using the same Python environment. Keep the project root available on `PYTHONPATH` (the client sets this for the child process).

## Run the CLI

Local tools:

```bash
python enhance.py samples/dim_room.jpg --output outputs/dim_room_fixed.jpg --no-mcp
```

MCP-backed tools:

```bash
python enhance.py samples/dim_room.jpg --output outputs/dim_room_mcp.jpg --mcp
```

The CLI prints the output image path, report path, quality score, run ID, latency, tool-call count and cost status. A JSON run record is written to `runs/<run_id>.json`.

## Run the API

```bash
export LOWLIGHT_USE_MCP=1
uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/` for the web page and `http://127.0.0.1:8000/docs` for interactive API docs.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health status and MCP mode |
| `/enhance` | POST | Upload an image and receive the enhanced image; the `X-Run-ID` header identifies the run |
| `/metrics/{run_id}` | GET | Read the structured JSON run record |
| `/evaluation/run` | POST | Run the 42-case evaluation suite |
| `/evaluation/results` | GET | Read the latest evaluation summary and per-case results |

Example upload:

```bash
curl -X POST 'http://127.0.0.1:8000/enhance?max_reflections=1' \
  -F 'file=@samples/dim_room.jpg' -o outputs/api_enhanced.jpg -D outputs/response_headers.txt
```

Find the run ID in `outputs/response_headers.txt`, then request `/metrics/<run_id>`.

## Evaluation and baseline experiments

Run the full suite with local tools:

```bash
python -m evaluation.run_suite
```

Run the suite through MCP:

```bash
python -m evaluation.run_suite --mcp
```

The suite includes 42 repeatable scenarios built from the three included sample images plus controlled invalid-input cases. Each valid-image scenario is run twice (reflection limit 0 and 1), producing 78 execution records in total. Positive cases include original images, underexposure, synthetic noise, low contrast, blur, resize degradation, brighter gamma transformation and mixed dark/noisy variants. Negative cases verify that missing/corrupt images fail in a controlled way.

The baseline applies fixed gamma correction (`gamma=0.72`) followed by non-local means denoising (`strength=7`). The agentic system is scored on the same transformed input. Results are saved to:

- `results/evaluation_results.json`
- `results/evaluation_results.csv`

Reported metrics include task success rate, mean latency, P95 latency, MCP/local tool success rate, average quality score, baseline quality score, and available cost information. The quality score is the project's own deterministic metric; it is not a human perceptual study or a universal image-quality benchmark. For a stronger submission, inspect sample outputs and record a small human review as well.

### Ablation experiment

The runner performs a paired ablation: every valid transformed input is processed once with `max_reflections=0` and once with `max_reflections=1`. Compare the paired per-case scores and latency in the CSV and the `reflection_ablation` summary. A second ablation can compare MCP-backed execution with local registry execution; these should produce comparable image results, while latency may differ due to the client/server boundary.

## Token usage and cost accounting

When a Gemini planning call succeeds, the run log stores input/output token counts returned by the provider. The app does **not** guess token counts from text length. Configure `GEMINI_INPUT_PRICE_PER_MILLION` and `GEMINI_OUTPUT_PRICE_PER_MILLION` only after checking official pricing for the exact model and recording the date in your report. If rates are not configured, the run record leaves `estimated_cost_usd` empty and explains why in `cost_status`. If the heuristic planner is used and no LLM call succeeds, API cost is recorded as zero for that run.

The report should distinguish variable model/API cost from fixed hosting or infrastructure cost. Do not present hosting cost as a precise per-query cost unless you have a defensible allocation method.

## Tests

```bash
pytest -q
```

Tests cover metric aggregation, malformed MCP image payloads, and parameter validation. The existing pipeline tests continue to cover the image-processing workflow. Run `pip install -r requirements.txt` before testing the real MCP transport.

## Gemini API setup

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to your key. Never commit `.env` or paste the key into source code.
3. Keep `GEMINI_MODEL=gemini-3.5-flash-lite` unless you choose another model currently available to your API key.
4. Run the CLI again. A successful LLM plan is labelled `gemini+rag`; if the key is missing or the API call fails, the run is labelled `heuristic-fallback`.

The app calls the Gemini Developer API through Google's official `google-genai` Python SDK. This is separate from the ChatGPT web app and does not require Anthropic.

## Deploy to Render (Docker)

1. Push this project folder to a GitHub repository without `.env`, `.venv`, or API keys.
2. In Render, create **New → Web Service** and connect that repository. Select the Docker runtime (the repository includes a `Dockerfile`).
3. Add environment variables in Render: `GEMINI_API_KEY` (your secret key), `GEMINI_MODEL=gemini-3.5-flash-lite`, `LOWLIGHT_USE_MCP=1`, and `LOWLIGHT_RUNS_DIR=runs`.
4. Deploy and open `https://YOUR-SERVICE.onrender.com/health`, then `/docs` to try the API.
5. In `/docs`, use `POST /enhance` or `POST /query`, select an image file, and execute. The response is the enhanced image; inspect `X-Run-ID`, `X-Final-Score`, and `X-Latency-MS` response headers. Use `GET /metrics/{run_id}` for run details.

Render's filesystem is ephemeral by default, so run records and evaluation results may not persist across redeploys or instance restarts unless a persistent disk or external storage is configured. Free services may also sleep when idle. Local MCP stdio starts the MCP server as a subprocess inside the same service container; do not treat it as a remotely accessible MCP endpoint.

## Deployment

```bash
docker build -t lowlight-agents .
docker run --rm -p 8000:8000 -e LOWLIGHT_USE_MCP=1 lowlight-agents
```

For a hosted deployment, set `LOWLIGHT_USE_MCP=1` and any API key/rate variables in the hosting provider's environment settings. Keep secrets out of source control. Check `/health` after deployment and run a small smoke test through `/enhance` before sharing the URL. The existing hosted URL may continue to serve the earlier deployment until this updated version is pushed and redeployed.

## Known limitations

- The MCP transport is local stdio; the server is launched as a child process. For multi-instance or remote deployment, configure a supported remote MCP transport and deployment security separately.
- Cost is only calculated when the provider reports tokens and exact model pricing rates have been configured.
- The current evaluator uses deterministic image metrics. It does not prove that every enhancement looks better to every person.
- The included evaluation suite is a reproducible student benchmark, not a large public dataset or a clinical/surveillance validation study.
- Synthetic image degradations are useful for controlled testing but do not represent every real camera condition.

## Future improvements

- Add a human-rated image quality subset and a public benchmark dataset.
- Add a remote MCP transport with authentication and explicit request timeouts for hosted deployments.
- Add structured traces for individual agent stages and a dashboard for latency/quality trade-offs.
- Compare against a learned low-light enhancement model when suitable weights and evaluation data are available.

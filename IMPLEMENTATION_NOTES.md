# Implementation and Verification Notes

This package preserves the existing low-light enhancement project and adds MCP integration, run telemetry, a workflow-level evaluator, a 42-scenario/78-execution evaluation suite, API metrics endpoints, and documentation.

## Verification completed in the build environment

- Python compilation: passed.
- Unit/pipeline tests: **23 passed, 1 skipped**.
- Local-tool evaluation: **78/78 executions passed**, including six expected controlled invalid-input rejections.
- Local-tool baseline comparison and reflection ablation results are included under `results/`.

## Important MCP verification limitation

The build environment did not have the MCP SDK installed and could not reach the package index to install it. Therefore, the real stdio MCP integration test was skipped here. The server/client implementation and dependency are included, but you should verify the MCP mode after installing dependencies on your Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python enhance.py samples/dim_room.jpg --output outputs/dim_room_mcp.jpg --mcp
```

The included evaluation results were produced with local tools (`mcp_enabled: false`). They must not be presented as MCP-mode benchmark results. After the MCP smoke test passes, run `python -m evaluation.run_suite --mcp` to create MCP-mode results.

## Cost reporting

The default heuristic planner makes no LLM API calls, so its model API cost is zero. If the Gemini planner is used, token counts are taken from provider-reported usage. Configure the exact model's current input/output rates before claiming a dollar cost; otherwise the run record reports that cost could not be calculated.


## Gemini and hosting update

- Anthropic was removed; the optional LLM planner now uses the official `google-genai` SDK and `GEMINI_API_KEY`.
- `/query` is an alias for the image-upload workflow at `/enhance`.
- Docker now honours Render's `PORT` environment variable.
- The MCP SDK is pinned below v2 to match the project's `mcp.server.fastmcp` imports.
- A live Gemini API request and a Render deployment must still be tested with the user's own API key and hosting account.

- Default Gemini model is `gemini-3.5-flash-lite`, selected as a lower-cost stable Flash option; model availability and current pricing should be checked in Google AI Studio before production use.

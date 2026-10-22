"""
Small tool-execution layer.

Agents don't call vision functions directly - they go through a registry
by name, the same way you'd expose functions to an LLM for tool-calling.
Doing it this way even for the non-LLM path buys us two things: a single
place that logs every operation performed on an image (handy for the run
report / debugging "why does this look weird"), and a consistent surface
that the LLM-backed planner can eventually call into directly if we want
it choosing tools itself rather than just parameters.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolCallRecord:
    tool: str
    ok: bool
    duration_ms: float
    error: str = ""


@dataclass
class ToolSpec:
    name: str
    func: Callable
    description: str


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self.call_log: list[ToolCallRecord] = []

    def register(self, name: str, func: Callable, description: str = "") -> None:
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        self._tools[name] = ToolSpec(name=name, func=func, description=description)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe(self, name: str) -> str:
        return self._tools[name].description

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"no tool registered under '{name}'")

        start = time.perf_counter()
        try:
            result = self._tools[name].func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is the tool boundary
            elapsed = (time.perf_counter() - start) * 1000
            self.call_log.append(ToolCallRecord(tool=name, ok=False, duration_ms=elapsed, error=str(exc)))
            raise
        else:
            elapsed = (time.perf_counter() - start) * 1000
            self.call_log.append(ToolCallRecord(tool=name, ok=True, duration_ms=elapsed))
            return result

    def recent_calls(self, n: int = 10) -> list[ToolCallRecord]:
        return self.call_log[-n:]

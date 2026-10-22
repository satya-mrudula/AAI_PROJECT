"""Persistent synchronous adapter around the official MCP Python SDK stdio client."""
import asyncio
import base64
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolClient:
    def __init__(self, timeout_seconds: float = 60):
        self.timeout_seconds = timeout_seconds
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop_worker, daemon=True)
        self._thread.start()
        try:
            self._future(self._connect()).result(timeout=timeout_seconds)
        except Exception as exc:
            try:
                if hasattr(self, "_stack"):
                    self._future(self._stack.aclose()).result(timeout=3)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            raise RuntimeError(f"Could not connect to the local MCP server: {exc}") from exc

    def _loop_worker(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _future(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _connect(self):
        self._stack = AsyncExitStack()
        server_path = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"
        project_root = str(server_path.parents[1])
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": project_root + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def _list_tools(self):
        return await self._session.list_tools()

    def list_tools(self):
        result = self._future(self._list_tools()).result(timeout=self.timeout_seconds)
        return sorted(tool.name for tool in result.tools)

    async def _call(self, name: str, arguments: dict):
        return await self._session.call_tool(name, arguments=arguments)

    def call_tool(self, name: str, arguments: dict):
        result = self._future(self._call(name, arguments)).result(timeout=self.timeout_seconds)
        if getattr(result, "isError", False):
            details = " ".join(getattr(item, "text", "") for item in getattr(result, "content", []))
            raise RuntimeError(f"MCP tool {name} failed: {details or 'unknown server error'}")
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        for item in getattr(result, "content", []):
            if getattr(item, "type", None) == "text":
                try:
                    return json.loads(item.text)
                except (ValueError, TypeError):
                    return item.text
        return None

    def close(self):
        if getattr(self, "_session", None) is not None:
            try:
                self._future(self._stack.aclose()).result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2)


def image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image for MCP tool call")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def base64_to_image(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("MCP tool returned an invalid image")
    return image

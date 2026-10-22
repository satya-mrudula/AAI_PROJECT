"""ToolRegistry-compatible adapter that routes image operations through MCP."""
import time

from mcp_client.client import MCPToolClient, image_to_base64, base64_to_image
from toolbox.registry import ToolRegistry, ToolCallRecord


_POSITIONAL_NAMES = {
    "apply_gamma": ["gamma"],
    "apply_clahe": ["clip_limit", "tile_grid"],
    "denoise_nlmeans": ["strength"],
    "denoise_bilateral": ["diameter", "sigma_color", "sigma_space"],
    "denoise_gaussian": ["ksize"],
    "unsharp_mask": ["amount"],
}
_IMAGE_RESULT_TOOLS = {
    "apply_gamma", "apply_clahe", "denoise_nlmeans", "denoise_bilateral",
    "denoise_gaussian", "unsharp_mask",
}


class MCPToolRegistry(ToolRegistry):
    """Same interface as the local registry, with a real MCP client underneath."""
    def __init__(self, local_registry: ToolRegistry, timeout_seconds: float = 60):
        super().__init__()
        self._tools = dict(local_registry._tools)
        self.client = MCPToolClient(timeout_seconds=timeout_seconds)
        self.available_mcp_tools = self.client.list_tools()
        required = set(self._tools)
        missing = sorted(required - set(self.available_mcp_tools))
        if missing:
            self.client.close()
            raise RuntimeError(f"MCP server is missing required tools: {', '.join(missing)}")

    def execute(self, name, *args, **kwargs):
        if name not in self._tools:
            raise KeyError(f"no tool registered under '{name}'")
        start = time.perf_counter()
        try:
            if name == "evaluate_quality":
                if len(args) != 2:
                    raise ValueError("evaluate_quality requires original and enhanced images")
                arguments = {"original_base64": image_to_base64(args[0]), "enhanced_base64": image_to_base64(args[1])}
            else:
                if not args:
                    raise ValueError(f"MCP image tool '{name}' requires an image argument")
                image = args[0]
                arguments = {"image_base64": image_to_base64(image)}
                names = _POSITIONAL_NAMES.get(name, [])
                for index, value in enumerate(args[1:]):
                    if index >= len(names):
                        raise TypeError(f"unexpected positional argument for MCP tool '{name}'")
                    arguments[names[index]] = value
                arguments.update(kwargs)
            result = self.client.call_tool(name, arguments)
            if isinstance(result, dict) and set(result.keys()) == {"result"}:
                result = result["result"]
            if name in _IMAGE_RESULT_TOOLS:
                result = base64_to_image(result["image_base64"])
            elapsed = (time.perf_counter() - start) * 1000
            self.call_log.append(ToolCallRecord(tool=name, ok=True, duration_ms=elapsed))
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            self.call_log.append(ToolCallRecord(tool=name, ok=False, duration_ms=elapsed, error=str(exc)))
            raise

    def close(self):
        self.client.close()

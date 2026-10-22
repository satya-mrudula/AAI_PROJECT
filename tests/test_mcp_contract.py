import importlib.util
import pytest

from mcp_server import tools


def test_mcp_tools_reject_malformed_image_payload():
    with pytest.raises(ValueError, match="base64"):
        tools.brightness_stats("not-valid-base64")


def test_gamma_tool_validates_range():
    import base64
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.zeros((12, 12, 3), dtype=np.uint8))
    assert ok
    payload = base64.b64encode(buf.tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="gamma"):
        tools.apply_gamma(payload, 99)


def test_mcp_server_exposes_required_tools_when_sdk_is_installed():
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("Install requirements.txt to run the real MCP client/server integration tests")
    from mcp_client.client import MCPToolClient
    client = MCPToolClient()
    try:
        available = set(client.list_tools())
        assert {"brightness_stats", "apply_gamma", "denoise_nlmeans", "evaluate_quality"} <= available
    finally:
        client.close()

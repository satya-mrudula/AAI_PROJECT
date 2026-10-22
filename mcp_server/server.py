"""Run with: python -m mcp_server.server
The MCP server exposes deterministic image capabilities; orchestration stays in the app.
"""
from mcp.server.fastmcp import FastMCP
from mcp_server import tools

mcp = FastMCP("lowlight-image-tools")

mcp.tool()(tools.brightness_stats)
mcp.tool()(tools.estimate_noise)
mcp.tool()(tools.apply_gamma)
mcp.tool()(tools.apply_clahe)
mcp.tool()(tools.denoise_nlmeans)
mcp.tool()(tools.denoise_bilateral)
mcp.tool()(tools.denoise_gaussian)
mcp.tool()(tools.unsharp_mask)
mcp.tool()(tools.evaluate_quality)

if __name__ == "__main__":
    mcp.run(transport="stdio")

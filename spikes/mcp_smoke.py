"""Smoke test: talk to semsearch-mcp over real stdio, like Claude Code does.

Run: uv run python spikes/mcp_smoke.py <project_root> "<query>"
"""
from __future__ import annotations

import shutil
import sys

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main(root: str, query: str) -> None:
    exe = shutil.which("semsearch-mcp")
    if exe is None:
        raise SystemExit("semsearch-mcp not on PATH (run inside `uv run`)")
    params = StdioServerParameters(command=exe, cwd=root)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        print("instructions present:", bool(session.instructions))
        result = await session.call_tool("semantic_search", {"query": query, "k": 3})
        for block in getattr(result, "content", []):
            print(getattr(block, "text", block))


if __name__ == "__main__":
    anyio.run(main, sys.argv[1], sys.argv[2])

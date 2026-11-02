"""MCP server (stdio) exposing semantic search over the current project to Claude Code."""
from __future__ import annotations

import logging
from pathlib import Path

import anyio
from mcp.server.mcpserver import MCPServer

from semsearch.client import DaemonClient, DaemonUnavailable
from semsearch.formatting import format_search, format_status
from semsearch.paths import find_project_root

log = logging.getLogger(__name__)

INSTRUCTIONS = """Semantic (meaning-based) search over this project's code and docs.
Use semantic_search when you know WHAT the code does but not its name ("where are failed
requests retried", "code that loads the player save file", "something similar to this snippet").
Do NOT use it for exact identifiers, strings, regexes, or when every occurrence is needed:
use Grep/Glob for those, and LSP (goToDefinition / findReferences / incomingCalls) for
definitions, references and call hierarchy. Results are ranked chunks with file:line ranges;
Read the lines you need before relying on them. When media indexing is enabled, results can
also be images (open them with Read), PDF pages ("file.pdf p.N": Read with pages=N) and
audio/video segments ("file 01:20-01:40": a time range; they cannot be opened with Read).
Write the query in English even when the user asked in another language: code is English and
translated queries find it much more reliably (benchmark: Chinese questions MRR 0.58 -> 0.70)."""

SEARCH_DESCRIPTION = """Find code/docs by meaning in the current project. Returns ranked
file:line ranges with symbol names and short previews. Best for intent-level questions where
the identifier is unknown. Write `query` in English (translate non-English questions first):
it ranks code far better. Not for exact names or exhaustive reference lists (use Grep / LSP)."""

server = MCPServer("semsearch", instructions=INSTRUCTIONS)
_client = DaemonClient()
_root: Path = find_project_root()


def _call(fn: str, *args: object, **kwargs: object) -> dict[str, object]:
    return getattr(_client, fn)(*args, **kwargs)


@server.tool(description=SEARCH_DESCRIPTION)
async def semantic_search(query: str, k: int = 8, path_glob: str | None = None,
                          modality: str | None = None) -> str:
    """query: natural-language description, in English; k: 1-50; path_glob: e.g. "src/**" or
    "*.cs"; modality: "code", "text", or (when media indexing is on) "image", "pdf", "audio",
    "video"."""
    globs = [g.strip() for g in path_glob.split(",") if g.strip()] if path_glob else None
    try:
        data = await anyio.to_thread.run_sync(
            lambda: _call("search", _root, query, k, globs, modality))
    except DaemonUnavailable as exc:
        return f"semantic search unavailable: {exc}. Fall back to Grep."
    return format_search(query, data)


@server.tool(description="Show semantic index status for the current project.")
async def index_status() -> str:
    try:
        data = await anyio.to_thread.run_sync(lambda: _call("status", _root))
    except DaemonUnavailable as exc:
        return f"semantic search unavailable: {exc}"
    return format_status(data)


@server.tool(description="Re-index the project: given paths only, or everything with full=true.")
async def reindex(paths: list[str] | None = None, full: bool = False) -> str:
    try:
        data = await anyio.to_thread.run_sync(lambda: _call("reindex", _root, full, paths))
    except DaemonUnavailable as exc:
        return f"semantic search unavailable: {exc}"
    if full:
        return "full re-index started"
    accepted = data.get("accepted")
    count = len(accepted) if isinstance(accepted, list) else 0
    return f"queued {count} paths" if paths else "rescan queued"


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per request is noise
    server.run("stdio")


if __name__ == "__main__":
    main()

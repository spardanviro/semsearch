"""MCP server (stdio) exposing semantic search over the current project to Claude Code."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

import anyio
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from semsearch.client import DaemonClient, DaemonRequestError, DaemonUnavailable
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
translated queries find it more reliably (in this project's benchmarks, translating Chinese
questions to English raised MRR from 0.77 to 0.80)."""

SEARCH_DESCRIPTION = """Find code/docs by meaning in the current project. Returns ranked
file:line ranges with symbol names and short previews. Best for intent-level questions where
the identifier is unknown. Write `query` in English (translate non-English questions first):
it ranks code far better. Not for exact names or exhaustive reference lists (use Grep / LSP)."""

server = MCPServer("semsearch", instructions=INSTRUCTIONS)
_client = DaemonClient()
_root: Path = find_project_root()


def _call(fn: str, *args: object, **kwargs: object) -> dict[str, object]:
    return getattr(_client, fn)(*args, **kwargs)


Modality = Literal["code", "text", "image", "pdf", "audio", "video"]


def split_globs(value: str) -> list[str]:
    """Comma-separated globs; commas inside {a,b} belong to the glob."""
    out, depth, current = [], 0, ""
    for ch in value:
        depth += (ch == "{") - (ch == "}")
        if ch == "," and depth <= 0:
            out.append(current)
            current, depth = "", 0
        else:
            current += ch
    out.append(current)
    return [g.strip() for g in out if g.strip()]


@server.tool(description=SEARCH_DESCRIPTION)
async def semantic_search(
        query: Annotated[str, Field(description="What the code does, in English")],
        k: Annotated[int, Field(ge=1, le=50, description="Number of results")] = 8,
        path_glob: Annotated[str | None, Field(description=(
            'Limit to paths, gitignore-style, comma-separated: "src/**", "*.cs", '
            '"**/*.{ts,tsx}", "!tests/**"'))] = None,
        modality: Annotated[Modality | None, Field(description=(
            'Only "code" or "text" (docs/config); "image", "pdf", "audio", "video" when '
            "media indexing is on"))] = None) -> str:
    globs = split_globs(path_glob) if path_glob else None
    try:
        data = await anyio.to_thread.run_sync(
            lambda: _call("search", _root, query, k, globs, modality))
    except DaemonRequestError as exc:
        return f"semantic search rejected the arguments: {exc}"
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

"""Markdown: one chunk per heading section, keeping the heading path as the symbol."""
from __future__ import annotations

import re

from semsearch.index.chunkers.base import Chunk, ChunkContext, non_blank
from semsearch.index.chunkers.window import window_chunks

# closing hashes only count after whitespace: "## Learn C#" keeps its "#"
HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def _sections(lines: list[str]) -> list[tuple[int, int, str]]:
    """-> [(start, end, heading_path)] with 1-based inclusive lines."""
    stack: list[tuple[int, str]] = []
    starts: list[tuple[int, str]] = []
    fence: str | None = None      # the opening marker: a fence closes only with the same
    for i, line in enumerate(lines, 1):     # character and at least the same length
        f = FENCE.match(line)
        if f:
            marker = f.group(1)
            if fence is None:
                fence = marker
                continue
            if marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
                continue
        m = None if fence is not None else HEADING.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        starts.append((i, " > ".join(t for _, t in stack)))
    if not starts or starts[0][0] != 1:
        starts.insert(0, (1, "(preamble)"))
    bounds = [s for s, _ in starts[1:]] + [len(lines) + 1]
    return [(s, nxt - 1, title) for (s, title), nxt in zip(starts, bounds, strict=True)]


def markdown_chunks(ctx: ChunkContext, lines: list[str]) -> list[Chunk]:
    out: list[Chunk] = []
    for start, end, title in _sections(lines):
        if non_blank(lines, start, end) == 0:
            continue
        chunk = ctx.build(lines, start, end, title, "section")
        if chunk.tokens <= ctx.target_tokens:
            out.append(chunk)
        else:
            out.extend(window_chunks(ctx, lines, start, end, title, "section"))
    return out

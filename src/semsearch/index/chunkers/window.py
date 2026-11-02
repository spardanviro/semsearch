"""Line-window chunking: the fallback for any text and for oversized definitions."""
from __future__ import annotations

from semsearch.index.chunkers.base import Chunk, ChunkContext, non_blank

BLANK_LOOKBACK = 10
MIN_NON_BLANK = 1


def _prefer_blank_break(lines: list[str], start: int, end: int) -> int:
    """Move `end` back to just before a blank line if one is close, so windows end cleanly."""
    for candidate in range(end, max(start + 1, end - BLANK_LOOKBACK), -1):
        if not lines[candidate - 1].strip():
            return candidate
    return end


def window_chunks(ctx: ChunkContext, lines: list[str], start: int, end: int,
                  symbol: str, kind: str) -> list[Chunk]:
    """Split 1-based inclusive [start, end] into overlapping windows within the token limit."""
    out: list[Chunk] = []
    pos = start
    while pos <= end:
        size = ctx.window_lines
        while True:
            stop = min(end, pos + size - 1)
            if stop < end:
                stop = _prefer_blank_break(lines, pos, stop)
            chunk = ctx.build(lines, pos, stop, symbol, kind)
            if chunk.tokens <= ctx.max_tokens or stop == pos:
                break
            size = max(1, (stop - pos + 1) // 2)
        if non_blank(lines, pos, stop) >= MIN_NON_BLANK:
            out.append(chunk)
        if stop >= end:
            break
        overlap = min(ctx.window_overlap, (stop - pos) // 2)
        pos = stop + 1 - overlap
    return out

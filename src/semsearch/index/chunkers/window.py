"""Line-window chunking: the fallback for any text and for oversized definitions."""
from __future__ import annotations

from semsearch.index.chunkers.base import Chunk, ChunkContext, non_blank

BLANK_LOOKBACK = 10
MIN_NON_BLANK = 1
MIN_PIECE_CHARS = 64


def _prefer_blank_break(lines: list[str], start: int, end: int) -> int:
    """Move `end` back to just before a blank line if one is close, so windows end cleanly."""
    for candidate in range(end, max(start + 1, end - BLANK_LOOKBACK), -1):
        if not lines[candidate - 1].strip():
            return candidate
    return end


def _split_line(ctx: ChunkContext, line: str, number: int, symbol: str,
                kind: str) -> list[Chunk]:
    """One line longer than max_tokens (minified code, inline data): cut it by characters,
    otherwise everything after the first ~max_tokens would be invisible to the embedding."""
    out: list[Chunk] = []
    pos = 0
    tokens = max(1, ctx.count_tokens(line))
    step = max(MIN_PIECE_CHARS, int(len(line) / tokens * ctx.max_tokens * 0.8))
    while pos < len(line):
        piece = line[pos:pos + step]
        chunk = ctx.build_body(piece, number, number, symbol, kind)
        while chunk.tokens > ctx.max_tokens and len(piece) > MIN_PIECE_CHARS:
            piece = piece[:len(piece) // 2]
            chunk = ctx.build_body(piece, number, number, symbol, kind)
        if piece.strip():
            out.append(chunk)
        pos += len(piece)
    return out


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
        if chunk.tokens > ctx.max_tokens:            # a single over-long line
            out.extend(_split_line(ctx, lines[pos - 1], pos, symbol, kind))
        elif non_blank(lines, pos, stop) >= MIN_NON_BLANK:
            out.append(chunk)
        if stop >= end:
            break
        overlap = min(ctx.window_overlap, (stop - pos) // 2)
        pos = max(stop + 1 - overlap, pos + 1)
    return out

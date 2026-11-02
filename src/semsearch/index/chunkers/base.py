"""Shared chunk type and helpers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    symbol: str
    kind: str
    text: str            # what gets embedded (header + body)
    preview: str         # what is shown to Claude
    tokens: int


HEADER_END = "\n---\n"


def make_header(path: str, symbol: str, kind: str, lang: str) -> str:
    return f"path: {path}\nsymbol: {symbol}\nkind: {kind}\nlang: {lang}{HEADER_END}"


def chunk_body(text: str) -> str:
    """`Chunk.text` without the header added by `make_header`."""
    _, sep, body = text.partition(HEADER_END)
    return body if sep else text


def make_preview(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body.rstrip()
    cut = body.rfind("\n", 0, limit)
    return body[: cut if cut > limit // 2 else limit].rstrip() + "\n…"


@dataclass(frozen=True)
class ChunkContext:
    path: str
    lang: str
    count_tokens: TokenCounter
    target_tokens: int
    max_tokens: int
    window_lines: int
    window_overlap: int
    preview_chars: int

    def build(self, lines: list[str], start: int, end: int, symbol: str, kind: str) -> Chunk:
        """Chunk from 1-based inclusive line range of `lines`."""
        body = "\n".join(lines[start - 1:end])
        text = make_header(self.path, symbol, kind, self.lang) + body
        return Chunk(start, end, symbol, kind, text, make_preview(body, self.preview_chars),
                     self.count_tokens(text))


def non_blank(lines: list[str], start: int, end: int) -> int:
    return sum(1 for ln in lines[start - 1:end] if ln.strip())

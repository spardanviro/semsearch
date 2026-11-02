"""Chunker dispatch: pick a strategy by modality / language."""
from __future__ import annotations

from pathlib import PurePosixPath

from semsearch.config import Settings
from semsearch.index.chunkers.base import Chunk, ChunkContext, TokenCounter
from semsearch.index.chunkers.code import code_chunks
from semsearch.index.chunkers.markdown import markdown_chunks
from semsearch.index.chunkers.window import window_chunks

__all__ = ["Chunk", "TokenCounter", "chunk_text", "looks_minified"]
MINIFIED_MIN_CHARS = 20_000
MINIFIED_AVG_LINE = 500


def looks_minified(text: str, line_count: int) -> bool:
    """Generated one-liners (app.min.js, bundled CSS, inline data): large, and very long lines
    of actual content (indentation does not count: deeply nested code is not minified)."""
    if len(text) <= MINIFIED_MIN_CHARS:
        return False
    content = len(text) - sum(text.count(ch) for ch in " \t\r\n")
    return content / max(1, line_count) > MINIFIED_AVG_LINE


def chunk_text(path: str, text: str, modality: str, lang: str,
               count_tokens: TokenCounter, settings: Settings) -> list[Chunk]:
    ctx = ChunkContext(path=path, lang=lang, count_tokens=count_tokens,
                       target_tokens=settings.target_tokens, max_tokens=settings.max_tokens,
                       window_lines=settings.window_lines, window_overlap=settings.window_overlap,
                       preview_chars=settings.preview_chars)
    if not text.strip():
        return []
    # split on "\n" only: tree-sitter counts rows the same way (splitlines() also breaks on
    # \x0c, \x85,   ... which would shift every line number after such a character)
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    if looks_minified(text, len(lines)):
        return []          # bundles / minified sources: noise for meaning-based search
    whole = ctx.build(lines, 1, len(lines), PurePosixPath(path).name, "file")
    if whole.tokens <= settings.target_tokens:
        return [whole]
    if modality == "code":
        chunks = code_chunks(ctx, lines, text.encode("utf-8"))
        if chunks:
            return chunks
    if lang == "markdown":
        return markdown_chunks(ctx, lines)
    return window_chunks(ctx, lines, 1, len(lines), PurePosixPath(path).name, "lines")

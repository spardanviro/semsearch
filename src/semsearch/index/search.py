"""Query -> ranked, de-duplicated chunk hits."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

import pathspec

from semsearch.embed.base import Embedder
from semsearch.index.store import ChunkHit, Store
from semsearch.index.vectors import VectorCache

CANDIDATE_FACTOR = 6
MAX_K = 50


@dataclass(frozen=True)
class SearchResult:
    path: str
    start_line: int | None
    end_line: int | None
    symbol: str
    kind: str
    modality: str
    score: float
    preview: str
    duplicates: tuple[str, ...] = ()   # other paths holding an identical chunk

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "lines": [self.start_line, self.end_line],
                "symbol": self.symbol, "kind": self.kind, "modality": self.modality,
                "score": round(self.score, 4), "preview": self.preview,
                "duplicates": list(self.duplicates)}


def path_filter_for(globs: list[str] | None) -> Callable[[str], bool] | None:
    if not globs:
        return None
    spec = pathspec.PathSpec.from_lines("gitignore", globs)
    return spec.match_file


def _overlaps(a: ChunkHit, b: SearchResult) -> bool:
    if a.start_line is None or b.start_line is None or a.end_line is None or b.end_line is None:
        return False
    return a.start_line <= b.end_line and b.start_line <= a.end_line


def search(store: Store, cache: VectorCache, embedder: Embedder, db_lock: threading.Lock,
           query: str, k: int = 8, path_globs: list[str] | None = None,
           modality: str | None = None, min_score: float | None = None,
           per_file: int = 2) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    k = max(1, min(k, MAX_K))
    [qvec] = embedder.encode_queries([query])
    candidates = cache.search(qvec, k * CANDIDATE_FACTOR, path_filter_for(path_globs), modality)
    with db_lock:
        meta = store.get_chunks([cid for cid, _ in candidates])

    results: list[SearchResult] = []
    per_path: dict[str, list[SearchResult]] = {}
    # identical code in copied trees -> one result (span length guards against truncated previews)
    by_content: dict[tuple[str, str, int], int] = {}
    for cid, score in candidates:
        hit = meta.get(cid)
        if hit is None:
            continue
        if min_score is not None and score < min_score:
            break
        span = (hit.end_line or 0) - (hit.start_line or 0)
        content_key = (hit.symbol, hit.preview, span)
        if content_key in by_content:
            i = by_content[content_key]
            if hit.path not in results[i].duplicates and hit.path != results[i].path:
                results[i] = replace(results[i], duplicates=(*results[i].duplicates, hit.path))
            continue
        same_file = per_path.setdefault(hit.path, [])
        if len(same_file) >= per_file or any(_overlaps(hit, r) for r in same_file):
            continue
        res = SearchResult(hit.path, hit.start_line, hit.end_line, hit.symbol, hit.kind,
                           hit.modality, score, hit.preview)
        same_file.append(res)
        by_content[content_key] = len(results)
        results.append(res)
        if len(results) >= k:
            break
    return results

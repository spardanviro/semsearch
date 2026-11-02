"""Query -> ranked, de-duplicated chunk hits.

Modes
- vector:  the default. Cosine similarity of EmbeddingGemma vectors (meaning, cross-language)
- keyword: FTS5 BM25 over identifier-split terms (exact names, rare words)
- hybrid:  questions with an identifier-shaped word get cosine + a small BM25 bonus
           (`boost`); plain-language questions are pure vector.
Displayed scores are always cosine similarity, so they mean the same thing in every mode.

Why (evals/, docs/PHASE1_5.md, docs/EVAL_REAL_PROJECTS.md): EmbeddingGemma already sees symbol
names through the chunk header. Hybrid helped a little only while doc comments were split
from their code; with chunker v4 it is level with vector over 126 cases (0.774 vs 0.778).
Rank fusion (RRF) and keyword hits on plain-language questions were worse.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from semsearch.config import DEFAULT_KEYWORD_WEIGHT, SEARCH_MODES
from semsearch.embed.base import Embedder
from semsearch.globs import path_filter
from semsearch.index import keywords
from semsearch.index.store import ChunkHit, Store
from semsearch.index.vectors import VectorCache

log = logging.getLogger(__name__)
CANDIDATE_FACTOR = 6
FILTERED_OVERFETCH = 4    # keyword hits are filtered after ranking: fetch more when filtering
MAX_KEYWORD_SCAN = 20_000  # upper bound on BM25 rows read for one filtered query
MAX_K = 50
MAX_POOL = 5000           # widest candidate pool when duplicates crowd the results
FTS_WEIGHTS = (2.0, 1.0)  # bm25 column weights: (path + symbol, body)
PathFilter = Callable[[str], bool]


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
    page: int | None = None            # PDF page
    t_start: float | None = None       # audio/video segment, seconds
    t_end: float | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "path": self.path, "lines": [self.start_line, self.end_line],
            "symbol": self.symbol, "kind": self.kind, "modality": self.modality,
            "score": round(self.score, 4), "preview": self.preview,
            "duplicates": list(self.duplicates)}
        if self.page is not None:
            out["page"] = self.page
        if self.t_start is not None:
            out["t_start"], out["t_end"] = self.t_start, self.t_end
        return out


def path_filter_for(globs: list[str] | None) -> PathFilter | None:
    """gitignore-style filter; normalised (./, backslashes, braces) and ReDoS-safe."""
    return path_filter(globs)


def _overlaps(a: ChunkHit, b: SearchResult) -> bool:
    if a.start_line is None or b.start_line is None or a.end_line is None or b.end_line is None:
        return False
    return a.start_line <= b.end_line and b.start_line <= a.end_line


def boost(vector: list[tuple[int, float]], keyword: list[tuple[int, float]],
          cosine: dict[int, float], weight: float) -> list[int]:
    """Cosine stays the main signal; a keyword hit adds up to `weight` (BM25 scaled to 0..1)."""
    top = keyword[0][1] if keyword and keyword[0][1] > 0 else 1.0
    bonus = {cid: weight * max(s, 0.0) / top for cid, s in keyword}
    ids = dict.fromkeys([cid for cid, _ in vector] + [cid for cid, _ in keyword])
    final = {cid: cosine.get(cid, -1.0) + bonus.get(cid, 0.0) for cid in ids}
    return sorted(final, key=lambda cid: -final[cid])


def _keyword_ranked(store: Store, db_lock: threading.Lock, query: str, n: int,
                    path_filter: PathFilter | None,
                    modality: str | None) -> list[tuple[int, float]]:
    match = keywords.match_expression(query)
    if match is None:
        return []
    if path_filter is None and modality is None:
        return _fts(store, db_lock, query, match, n)
    # filters run after BM25 ranking: widen the window until enough hits survive (or none left)
    limit = n * FILTERED_OVERFETCH
    while True:
        hits = _fts(store, db_lock, query, match, limit)
        with db_lock:
            meta = store.get_chunks([cid for cid, _ in hits])
        kept = [(cid, s) for cid, s in hits if (m := meta.get(cid)) is not None
                and (modality is None or m.modality == modality)
                and (path_filter is None or path_filter(m.path))]
        if len(kept) >= n or len(hits) < limit or limit >= MAX_KEYWORD_SCAN:
            return kept[:n]
        limit = min(limit * FILTERED_OVERFETCH, MAX_KEYWORD_SCAN)


def _fts(store: Store, db_lock: threading.Lock, query: str, match: str,
         limit: int) -> list[tuple[int, float]]:
    try:
        with db_lock:
            return store.keyword_search(match, limit, FTS_WEIGHTS)
    except sqlite3.Error:   # never let the keyword side break a search
        log.exception("keyword search failed for %r", query)
        return []


def _ranked_candidates(store: Store, cache: VectorCache, db_lock: threading.Lock, query: str,
                       qvec: np.ndarray, n: int, path_filter: PathFilter | None,
                       modality: str | None, mode: str,
                       keyword_weight: float) -> tuple[list[int], dict[int, float]]:
    """-> (chunk ids best first, cosine score per id)."""
    vector = cache.search(qvec, n, path_filter, modality) if mode != "keyword" else []
    cosine = dict(vector)
    # plain-language questions gain nothing from BM25 (benchmark): hybrid = vector for them
    if mode == "vector" or (mode == "hybrid" and not keywords.has_identifier(query)):
        return [cid for cid, _ in vector], cosine
    keyword = _keyword_ranked(store, db_lock, query, n, path_filter, modality)
    cosine.update(cache.scores(qvec, [cid for cid, _ in keyword if cid not in cosine]))
    if mode == "keyword":
        return [cid for cid, _ in keyword], cosine
    return boost(vector, keyword, cosine, keyword_weight), cosine


def search(store: Store, cache: VectorCache, embedder: Embedder, db_lock: threading.Lock,
           query: str, k: int = 8, path_globs: list[str] | None = None,
           modality: str | None = None, min_score: float | None = None,
           per_file: int = 2, mode: str = "vector",
           keyword_weight: float = DEFAULT_KEYWORD_WEIGHT) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
    k = max(1, min(k, MAX_K))
    [qvec] = embedder.encode_queries([query])
    match = path_filter_for(path_globs)
    pool = k * CANDIDATE_FACTOR
    while True:
        ranked, cosine = _ranked_candidates(store, cache, db_lock, query, qvec, pool, match,
                                            modality, mode, keyword_weight)
        with db_lock:
            meta = store.get_chunks(ranked)
        results = _collect(ranked, cosine, meta, k, min_score, per_file)
        # many identical copies (vendored trees) or one huge file can use up the whole pool:
        # widen it while results are missing and the pool was actually full
        if len(results) >= k or len(ranked) < pool or pool >= MAX_POOL:
            return results
        pool = min(pool * 4, MAX_POOL)


def _collect(ranked: list[int], cosine: dict[int, float], meta: dict[int, ChunkHit], k: int,
             min_score: float | None, per_file: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    per_path: dict[str, list[SearchResult]] = {}
    # identical code in copied trees -> one result (span length guards against truncated previews)
    by_content: dict[tuple[str, str, int], int] = {}
    for cid in ranked:
        hit = meta.get(cid)
        score = cosine.get(cid)
        if hit is None or score is None:
            continue
        if min_score is not None and score < min_score:
            continue
        span = (hit.end_line or 0) - (hit.start_line or 0)
        # text copies have equal symbol/preview/span; media previews ("[audio 00:00-00:30 of
        # 03:12]") repeat across different files, so media compares by content hash instead
        content_key = ((hit.symbol, hit.preview, span) if hit.start_line is not None
                       else ("media", hit.chunk_hash, 0))
        if content_key in by_content:
            i = by_content[content_key]
            if hit.path not in results[i].duplicates and hit.path != results[i].path:
                results[i] = replace(results[i], duplicates=(*results[i].duplicates, hit.path))
            continue
        same_file = per_path.setdefault(hit.path, [])
        if len(same_file) >= per_file or any(_overlaps(hit, r) for r in same_file):
            continue
        res = SearchResult(hit.path, hit.start_line, hit.end_line, hit.symbol, hit.kind,
                           hit.modality, score, hit.preview, page=hit.page,
                           t_start=hit.t_start, t_end=hit.t_end)
        same_file.append(res)
        by_content[content_key] = len(results)
        results.append(res)
        if len(results) >= k:
            break
    return results

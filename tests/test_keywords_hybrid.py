"""Keyword terms, the FTS5 index and hybrid (vector + keyword) ranking."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from semsearch.config import Settings
from semsearch.embed.base import normalize
from semsearch.index import keywords
from semsearch.index.search import boost, search
from semsearch.index.store import NewChunk, Store
from semsearch.index.vectors import VectorCache

DIM = 8


def vec(*values: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return normalize(v[None, :])[0]


# --- keywords --------------------------------------------------------------------------------
def test_keywords_terms_split_identifiers_and_keep_whole_name() -> None:
    assert keywords.terms("getPlayerInventory") == ["getplayerinventory", "get", "player", "inventory"]
    assert keywords.terms("CREATE_BREAKAWAY_FROM_JOB") == [
        "create_breakaway_from_job", "create", "breakaway", "from", "job"]
    assert keywords.terms("HTTPServer x") == ["httpserver", "http", "server"]


def test_keywords_terms_ignore_cjk_and_single_letters() -> None:
    assert keywords.terms("存档 a SaveP2 损坏") == ["savep2", "save"]


@pytest.mark.parametrize(("query", "expected"), [
    ("TranspilerGuard Verify", True),
    ("occupyingPlayerID", True),
    ("CREATE_BREAKAWAY_FROM_JOB", True),
    ("BEGIN IMMEDIATE rollback", True),
    ("write_daemon_file 设置权限", True),
    ("restore the user's vsync setting", False),
    ("P2 存档先写临时文件", False),
    ("apply each Harmony patch", False),
    ("HTTPServer handler", True),          # regression: acronym-first PascalCase was missed
    ("XMLParser", True),
    ("the 3rd argument is 10px wide", False),   # regression: digit-first words triggered hybrid
])
def test_keywords_has_identifier_detects_code_shaped_words(query: str, expected: bool) -> None:
    assert keywords.has_identifier(query) is expected


def test_keywords_match_expression_drops_stopwords_and_quotes_terms() -> None:
    assert keywords.match_expression("where is the loadConfig") == '"loadconfig" OR "load" OR "config"'
    assert keywords.match_expression("how is the 存档") is None


# --- store -----------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "index.db")
    yield s
    s.close()


def chunk(h: str, v: np.ndarray, symbol: str, body: str, path: str = "a.py") -> NewChunk:
    return NewChunk(h, "code", 1, 3, symbol, "function_definition", body, 5, v,
                    keywords.document(f"{path} {symbol}"), keywords.document(body))


def test_store_keyword_search_ranks_matching_chunks(store: Store) -> None:
    fid = store.upsert_file("a.py", 1, 1, "h", "code", "python", 0.0)
    a, b = store.insert_chunks(fid, [chunk("1", vec(1), "start", "flags = CREATE_BREAKAWAY_FROM_JOB"),
                                     chunk("2", vec(0, 1), "stop", "shutdown the server")])

    hits = store.keyword_search('"create_breakaway_from_job"', 10, (2.0, 1.0))

    assert [cid for cid, _ in hits] == [a]
    assert hits[0][1] > 0
    assert b not in {cid for cid, _ in hits}


def test_store_deleting_file_and_clear_remove_keyword_rows(store: Store) -> None:
    fid = store.upsert_file("a.py", 1, 1, "h", "code", "python", 0.0)
    store.insert_chunks(fid, [chunk("1", vec(1), "uniqueName", "zebra")])
    fid2 = store.upsert_file("b.py", 1, 1, "h", "code", "python", 0.0)
    store.insert_chunks(fid2, [chunk("2", vec(1), "other", "zebra", "b.py")])

    store.delete_file("a.py")
    after_delete = store.keyword_search('"zebra"', 10, (2.0, 1.0))
    store.clear()

    assert len(after_delete) == 1
    assert store.keyword_search('"zebra"', 10, (2.0, 1.0)) == []


# --- ranking ---------------------------------------------------------------------------------
def test_boost_keeps_cosine_order_unless_keyword_bonus_outweighs_gap() -> None:
    vector = [(1, 0.80), (2, 0.79)]
    keyword = [(2, 9.0), (3, 3.0)]
    cosine = {1: 0.80, 2: 0.79, 3: 0.10}

    small = boost(vector, keyword, cosine, 0.005)
    large = boost(vector, keyword, cosine, 0.05)

    assert small[:2] == [1, 2]
    assert large[:2] == [2, 1]
    assert large[-1] == 3


class StubEmbedder:
    """Every query embeds to the same vector: the vector side always prefers chunk `near`."""
    model_id, dim, load_mode = "stub", DIM, "test"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([vec(1, 0)] * len(texts))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


@pytest.fixture
def indexed(store: Store) -> tuple[Store, VectorCache, dict[str, int]]:
    fid = store.upsert_file("src/near.py", 1, 1, "h", "code", "python", 0.0)
    fid2 = store.upsert_file("lib/far.py", 1, 1, "h", "code", "python", 0.0)
    rows = {"near": (fid, chunk("1", vec(1, 0.1), "near", "start the worker", "src/near.py")),
            "far": (fid2, chunk("2", vec(0.6, 1), "far", "flags = CREATE_BREAKAWAY_FROM_JOB",
                                "lib/far.py"))}
    cache = VectorCache(DIM)
    ids = {}
    for name, (file_id, c) in rows.items():
        [cid] = store.insert_chunks(file_id, [c])
        ids[name] = cid
        cache.add([(cid, "src/near.py" if name == "near" else "lib/far.py", "code", c.embedding)])
    return store, cache, ids


def run(indexed: tuple[Store, VectorCache, dict[str, int]], query: str, mode: str,
        **kw: object) -> list[str]:
    store, cache, _ = indexed
    results = search(store, cache, StubEmbedder(), threading.Lock(), query, 5, mode=mode,
                     **kw)  # type: ignore[arg-type]
    return [r.symbol for r in results]


def test_search_hybrid_lifts_exact_identifier_match(
        indexed: tuple[Store, VectorCache, dict[str, int]]) -> None:
    assert run(indexed, "CREATE_BREAKAWAY_FROM_JOB", "vector") == ["near", "far"]
    assert run(indexed, "CREATE_BREAKAWAY_FROM_JOB", "hybrid", keyword_weight=1.0) == ["far", "near"]
    assert run(indexed, "CREATE_BREAKAWAY_FROM_JOB", "keyword") == ["far"]


def test_search_hybrid_ignores_keywords_for_plain_questions(
        indexed: tuple[Store, VectorCache, dict[str, int]]) -> None:
    assert run(indexed, "create breakaway from job", "hybrid", keyword_weight=1.0) == ["near", "far"]


def test_search_keyword_hits_respect_path_filter_and_report_cosine(
        indexed: tuple[Store, VectorCache, dict[str, int]]) -> None:
    store, cache, _ = indexed

    filtered = search(store, cache, StubEmbedder(), threading.Lock(), "CREATE_BREAKAWAY_FROM_JOB",
                      5, path_globs=["src/**"], mode="keyword")  # type: ignore[arg-type]
    [far] = search(store, cache, StubEmbedder(), threading.Lock(), "CREATE_BREAKAWAY_FROM_JOB",
                   5, mode="keyword")  # type: ignore[arg-type]

    assert filtered == []
    assert far.score == pytest.approx(float(vec(1, 0) @ vec(0.6, 1)))


def test_search_narrow_filter_widens_keyword_window(store: Store,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: BM25 hits were cut to a fixed window before path filtering, so a narrow
    # glob could silently drop every keyword match.
    import semsearch.index.search as search_mod
    monkeypatch.setattr(search_mod, "FILTERED_OVERFETCH", 2)
    cache = VectorCache(DIM)
    for i in range(40):   # strong keyword matches outside the filter
        fid = store.upsert_file(f"other/f{i}.py", 1, 1, "h", "code", "python", 0.0)
        [cid] = store.insert_chunks(fid, [chunk(f"o{i}", vec(0, 1), "zebraHandler",
                                                "zebraHandler zebraHandler", f"other/f{i}.py")])
        cache.add([(cid, f"other/f{i}.py", "code", vec(0, 1))])
    fid = store.upsert_file("want/x.py", 1, 1, "h", "code", "python", 0.0)
    [want] = store.insert_chunks(fid, [chunk("w", vec(0, 1), "misc", "uses zebraHandler once "
                                             + "filler " * 50, "want/x.py")])
    cache.add([(want, "want/x.py", "code", vec(0, 1))])

    hits = search(store, cache, StubEmbedder(), threading.Lock(), "zebraHandler", 1,
                  path_globs=["want/**"], mode="keyword")  # type: ignore[arg-type]

    assert [r.path for r in hits] == ["want/x.py"]


@pytest.mark.parametrize("bad", ["nan", "inf", "-0.1"])
def test_settings_reject_invalid_keyword_weight(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SEMSEARCH_KEYWORD_WEIGHT", bad)

    with pytest.raises(ValueError, match="SEMSEARCH_KEYWORD_WEIGHT"):
        Settings()


def test_search_rejects_unknown_mode(indexed: tuple[Store, VectorCache, dict[str, int]]) -> None:
    with pytest.raises(ValueError, match="mode"):
        run(indexed, "x", "fuzzy")


def test_settings_reject_unknown_search_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMSEARCH_SEARCH_MODE", "fuzzy")

    with pytest.raises(ValueError, match="SEMSEARCH_SEARCH_MODE"):
        Settings()


def test_vector_cache_scores_specific_ids() -> None:
    cache = VectorCache(DIM)
    cache.add([(1, "a.py", "code", vec(1, 0)), (2, "b.py", "code", vec(0, 1))])

    scores = cache.scores(vec(1, 0), [2, 1, 99])

    assert scores == {1: pytest.approx(1.0), 2: pytest.approx(0.0)}
    assert cache.scores(vec(1, 0), [99]) == {}
